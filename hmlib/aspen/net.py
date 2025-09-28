import importlib
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

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
        self.minimal_context = bool(minimal_context or (isinstance(graph_cfg, dict) and graph_cfg.get("minimal_context", False)))

        # Accept either {trunks: {...}} or a flat dict
        trunks = graph_cfg.get("trunks") if isinstance(graph_cfg, dict) else None
        if trunks is None:
            raise ValueError("AspenNet expects a dict with a 'trunks' mapping.")

        self._build_nodes(trunks)
        self.exec_order = self._toposort()

    # region build
    def _build_nodes(self, trunks: Dict[str, Any]):
        for name, spec in trunks.items():
            if spec is None:
                raise ValueError(f"Empty spec for trunk '{name}'")
            cls_path = spec.get("class")
            if not cls_path:
                raise ValueError(f"Trunk '{name}' missing 'class'")
            depends = spec.get("depends", []) or []
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
        # Kahn's algorithm
        deps: Dict[str, Set[str]] = {}
        rev: Dict[str, Set[str]] = {}
        name2node: Dict[str, _Node] = {n.name: n for n in self.nodes}

        for n in self.nodes:
            deps[n.name] = set(n.depends)
            for d in n.depends:
                rev.setdefault(d, set()).add(n.name)

        q: deque[str] = deque([n.name for n in self.nodes if not deps[n.name]])
        order: List[str] = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in list(rev.get(u, [])):
                deps[v].discard(u)
                if not deps[v]:
                    q.append(v)
            rev.pop(u, None)

        if len(order) != len(self.nodes):
            remaining = {k: list(v) for k, v in deps.items() if v}
            raise ValueError(f"Cycle detected in trunks graph: {remaining}")

        return [name2node[n] for n in order]
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


class _NoOpTrunk(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self._name = name

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        # Intentionally does nothing
        return {}
