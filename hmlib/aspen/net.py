import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx
import torch


@dataclass
class _Node:
    name: str
    cls_path: str
    depends: List[str]
    params: Dict[str, Any]
    module: torch.nn.Module


class AspenNet(torch.nn.Module):
    """
    A configurable directed-acyclic graph runner for "trunks".

    - Loads a YAML-like dict (already parsed) with nodes definitions.
    - Each node has: name, class (import path), depends (list), and params.
    - Executes nodes in topological order, passing and accumulating a
      shared context dict across nodes.
    """

    def __init__(self, graph_cfg: Dict[str, Any], shared: Optional[Dict[str, Any]] = None, minimal_context: bool = False):
        super().__init__()
        self.shared: Dict[str, Any] = shared or {}
        self.nodes: List[_Node] = []
        # NetworkX DiGraph storing the trunks graph and attributes
        self.graph: nx.DiGraph = nx.DiGraph()
        self.minimal_context = bool(minimal_context or (isinstance(graph_cfg, dict) and graph_cfg.get("minimal_context", False)))

        # Accept either {trunks: {...}} or a flat dict
        trunks = graph_cfg.get("trunks") if isinstance(graph_cfg, dict) else None
        if trunks is None:
            raise ValueError("AspenNet expects a dict with a 'trunks' mapping.")

        self._build_nodes(trunks)
        self._build_graph(trunks)
        self.exec_order = self._toposort()
        self.training: bool = False
        # self.display_graphviz()

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.nodes:
            module.module.train(mode)
        return super().train(mode)

    def eval(self, mode: bool = True):
        self.training = not mode
        for module in self.nodes:
            module.module.eval(mode)
        return super().eval(mode)

    # region build
    def _build_nodes(self, trunks: Dict[str, Any]):
        for name, spec in trunks.items():
            if spec is None:
                raise ValueError(f"Empty spec for trunk '{name}'")
            cls_path = spec.get("class")
            if not cls_path:
                raise ValueError(f"Trunk '{name}' missing 'class'")
            depends = list(spec.get("depends", []) or [])
            params = spec.get("params", {}) or {}
            enabled = spec.get("enabled", True)
            if not enabled:
                # Create a no-op stub to keep graph shape predictable
                module = _NoOpTrunk(name=name)
            else:
                module = self._instantiate(cls_path, params)
            node = _Node(name=name, cls_path=cls_path, depends=depends, params=params, module=module)
            setattr(self, f"trunk_{name}", module)
            self.nodes.append(node)

    def _build_graph(self, trunks: Dict[str, Any]):
        # Add all nodes with attributes
        for node in self.nodes:
            self.graph.add_node(
                node.name,
                cls_path=node.cls_path,
                params=node.params,
                module=node.module,
            )

        # Add all edges (dep -> node)
        unknown_deps: Dict[str, List[str]] = {}
        all_names = {n.name for n in self.nodes}
        for node in self.nodes:
            for dep in node.depends:
                if dep not in all_names:
                    unknown_deps.setdefault(node.name, []).append(dep)
                    continue
                self.graph.add_edge(dep, node.name)

        if unknown_deps:
            details = ", ".join(f"{k}: {v}" for k, v in unknown_deps.items())
            raise ValueError(f"Unknown dependencies referenced in trunks: {details}")

    @staticmethod
    def _instantiate(cls_path: str, params: Dict[str, Any]) -> torch.nn.Module:
        mod_name, _, cls_name = cls_path.rpartition(".")
        if not mod_name:
            raise ValueError(f"Invalid class path '{cls_path}'")
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        if not issubclass(cls, torch.nn.Module):
            raise TypeError(f"Class '{cls_path}' must derive from torch.nn.Module")
        return cls(**params)

    def _toposort(self) -> List[_Node]:
        if not nx.is_directed_acyclic_graph(self.graph):
            try:
                cycle_nodes = nx.find_cycle(self.graph)  # type: ignore[arg-type]
            except Exception:
                cycle_nodes = []
            raise ValueError(f"Cycle detected in trunks graph: {cycle_nodes}")

        name2node: Dict[str, _Node] = {n.name: n for n in self.nodes}
        order_names: List[str] = list(nx.topological_sort(self.graph))
        return [name2node[n] for n in order_names]
    # endregion

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all trunks in topological order.

        - 'context' is a mutable dict; trunks can read and write.
        - Returns the final context for convenience.
        """
        # Ensure trunks can access shared resources
        context.setdefault("shared", self.shared)
        context.setdefault("trunks", {})
        with torch.no_grad() if not self.training else torch.enable_grad():
            for node in self.exec_order:
                trunk = node.module
                # Build a sub-context with only requested inputs, if enabled
                if self.minimal_context:
                    req = set(getattr(trunk, "input_keys", lambda: set())())
                    subctx: Dict[str, Any] = {}
                    # pull from local context first, then shared if missing
                    for k in req:
                        if k in context:
                            subctx[k] = context[k]
                        elif k in self.shared:
                            subctx[k] = self.shared[k]
                    # Provide shared for convenience if requested
                    subctx.setdefault("shared", self.shared)
                else:
                    subctx = context

                out = trunk(subctx)
                if not out:
                    out = {}

                # Determine which keys were modified
                declared = set(getattr(trunk, "output_keys", lambda: set())())
                update_keys = declared if declared else set(out.keys())

                # Merge into a next context: copy so trunks can delete safely
                for k in update_keys:
                    if k in out:
                        v = out[k]
                        from .trunks.base import DeleteKey  # local import avoids cycle

                        if isinstance(v, DeleteKey):
                            if k in context:
                                del context[k]
                        else:
                            context[k] = v

                # Store trunk-local outputs for introspection
                context["trunks"][node.name] = {k: out[k] for k in out.keys()}
        return context

    # region graph export/visualization
    def to_networkx(self) -> nx.DiGraph:
        """Return a shallow copy of the internal NetworkX DiGraph."""
        return self.graph.copy()

    def _dot_lines(self) -> Iterable[str]:
        # Simple DOT writer without extra deps
        yield "digraph AspenNet {"
        yield "  rankdir=LR;"
        yield "  node [shape=box, style=rounded];"
        # Nodes with labels
        for n, data in self.graph.nodes(data=True):
            label = f"{n}\n{data.get('cls_path', '')}"
            yield f'  "{n}" [label="{label}"];'
        # Edges
        for u, v in self.graph.edges():
            yield f'  "{u}" -> "{v}";'
        yield "}"

    def to_dot(self) -> str:
        """Return the Graphviz DOT string for the trunks graph."""
        return "\n".join(self._dot_lines())

    def save_graphviz(self, path: str) -> None:
        """
        Save the trunks graph as a Graphviz DOT file.

        Args:
            path: Destination file path (e.g., "graph.dot").
        """
        dot = self.to_dot()
        with open(path, "w", encoding="utf-8") as f:
            f.write(dot)

    def display_graphviz(self) -> None:
        """
        Display the trunks graph.

        Tries, in order:
        - graphviz.Source (if `graphviz` python package is installed)
        - matplotlib via networkx (if matplotlib is available)
        - Prints DOT to stdout as a fallback
        """
        dot = self.to_dot()

        # Try graphviz Python package
        try:
            from graphviz import Source  # type: ignore

            src = Source(dot)
            src.view(cleanup=True)
            return
        except Exception as ex:
            print(f"AspenNet: graphviz display failed: {ex}")

        # Try matplotlib networkx draw
        try:
            import matplotlib.pyplot as plt  # type: ignore

            pos = (
                nx.nx_agraph.graphviz_layout(self.graph, prog="dot")
                if self._has_pygraphviz()
                else nx.spring_layout(self.graph)
            )
            nx.draw(self.graph, pos, with_labels=True, node_size=1500, node_color="#DDEEFF", font_size=8, arrows=True)
            plt.title("AspenNet Trunks Graph")
            plt.show()
            return
        except Exception as e:
            print(f"AspenNet: matplotlib display failed: {e}")

        # Fallback: print DOT to stdout
        print(dot)

    @staticmethod
    def _has_pygraphviz() -> bool:
        try:
            import pygraphviz  # type: ignore  # noqa: F401

            return True
        except Exception:
            return False

    # endregion


class _NoOpTrunk(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self._name = name

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        # Intentionally does nothing
        return {}
