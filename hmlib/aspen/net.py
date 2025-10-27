import importlib
import contextlib
import os
import threading
from dataclasses import dataclass
from queue import Queue
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
        pipeline_cfg: Dict[str, Any] = {}
        if isinstance(graph_cfg, dict):
            pipeline_cfg = graph_cfg.get("pipeline", {}) or {}
        if not isinstance(pipeline_cfg, dict):
            raise ValueError("AspenNet 'pipeline' configuration must be a mapping if provided.")
        threaded_flag = pipeline_cfg.get("threaded")
        if threaded_flag is None and isinstance(graph_cfg, dict):
            threaded_flag = graph_cfg.get("threaded_trunks", False)
        self.threaded_trunks: bool = bool(threaded_flag)
        queue_size_cfg = pipeline_cfg.get("queue_size", 1)
        try:
            self.thread_queue_size: int = max(1, int(queue_size_cfg))
        except Exception as exc:
            raise ValueError(f"AspenNet pipeline queue_size must be an integer, got {queue_size_cfg!r}") from exc
        cuda_streams_flag = pipeline_cfg.get("cuda_streams", True)
        self.thread_cuda_streams: bool = bool(cuda_streams_flag)

        # Accept either {trunks: {...}} or a flat dict
        trunks = graph_cfg.get("trunks") if isinstance(graph_cfg, dict) else None
        if trunks is None:
            raise ValueError("AspenNet expects a dict with a 'trunks' mapping.")

        # Profiler wiring (optional and zero-overhead when absent)
        self._profiler = self.shared.get("profiler", None)

        self._build_nodes(trunks)
        self._build_graph(trunks)
        self.exec_order = self._toposort()
        self.training: bool = False
        self.save_graphviz("aspennet.dot")

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
        if self.threaded_trunks:
            return self._forward_threaded(context)
        grad_ctx = torch.enable_grad() if self.training else torch.no_grad()
        with grad_ctx:
            for node in self.exec_order:
                self._execute_node(node, context)
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

    def _execute_node(self, node: _Node, context: Dict[str, Any]) -> None:
        trunk = node.module
        subctx = self._make_subcontext(trunk, context) if self.minimal_context else context
        if getattr(self._profiler, "enabled", False):
            name = f"aspen.trunk.{node.name}"
            with self._profiler.rf(name):
                out = trunk(subctx) or {}
        else:
            out = trunk(subctx) or {}

        declared = set(getattr(trunk, "output_keys", lambda: set())())
        update_keys = declared if declared else set(out.keys())

        from .trunks.base import DeleteKey  # local import avoids cycle

        for key in update_keys:
            if key in out:
                value = out[key]
                if isinstance(value, DeleteKey):
                    if key in context:
                        del context[key]
                else:
                    context[key] = value

        context["trunks"][node.name] = {k: out[k] for k in out.keys()}

    def _make_subcontext(self, trunk: torch.nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        req_keys = set(getattr(trunk, "input_keys", lambda: set())())
        if not req_keys:
            subctx: Dict[str, Any] = {}
        else:
            subctx = {k: context[k] for k in req_keys if k in context}
            for key in req_keys:
                if key not in subctx and key in self.shared:
                    subctx[key] = self.shared[key]
        subctx.setdefault("shared", self.shared)
        return subctx

    def _forward_threaded(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stop_token = object()

        class _ExceptionWrapper:
            __slots__ = ("exc", "tb")

            def __init__(self, exc: BaseException):
                self.exc = exc
                self.tb = exc.__traceback__

            def reraise(self) -> None:
                raise self.exc.with_traceback(self.tb)

        def make_grad_ctx():
            return torch.enable_grad() if self.training else torch.no_grad()

        def worker(index: int, node: _Node) -> None:
            in_queue = queues[index]
            out_queue = queues[index + 1]
            while True:
                item = in_queue.get()
                if item is stop_token:
                    out_queue.put(stop_token)
                    break
                if isinstance(item, _ExceptionWrapper):
                    out_queue.put(item)
                    break
                grad_ctx = make_grad_ctx()
                try:
                    with grad_ctx:
                        self._execute_with_stream(node, item)
                    out_queue.put(item)
                except BaseException as exc:
                    out_queue.put(_ExceptionWrapper(exc))
                    break

        queues: List[Queue] = [Queue(maxsize=self.thread_queue_size) for _ in range(len(self.exec_order) + 1)]
        threads = []
        for idx, node in enumerate(self.exec_order):
            thread = threading.Thread(target=worker, args=(idx, node), daemon=True, name=node.name)
            thread.start()
            threads.append(thread)

        queues[0].put(context)
        result = queues[-1].get()
        try:
            if isinstance(result, _ExceptionWrapper):
                result.reraise()
            return result
        finally:
            if threads:
                queues[0].put(stop_token)
            for thread in threads:
                thread.join()

    def _execute_with_stream(self, node: _Node, context: Dict[str, Any]) -> None:
        device = self._infer_device(context)
        use_cuda_stream = self.thread_cuda_streams and torch.cuda.is_available() and device is not None
        if use_cuda_stream:
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                self._execute_node(node, context)
            stream.synchronize()
        else:
            self._execute_node(node, context)

    def _infer_device(self, context: Dict[str, Any]) -> Optional[torch.device]:
        device = context.get("device")
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            try:
                return torch.device(device)
            except Exception:
                pass
        cuda_stream = context.get("cuda_stream")
        if hasattr(cuda_stream, "device"):
            try:
                return torch.device(cuda_stream.device)  # type: ignore[arg-type]
            except Exception:
                pass
        data = context.get("data")
        if isinstance(data, dict):
            img = data.get("img")
            if isinstance(img, torch.Tensor):
                return img.device
        shared_device = self.shared.get("device")
        if isinstance(shared_device, torch.device):
            return shared_device
        if isinstance(shared_device, str):
            try:
                return torch.device(shared_device)
            except Exception:
                pass
        if torch.cuda.is_available():
            try:
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            except Exception:
                pass
        return None


class _NoOpTrunk(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self._name = name

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        # Intentionally does nothing
        return {}
