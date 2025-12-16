"""Execution engine for Aspen plugin graphs.

Constructs a directed acyclic graph of plugins, then runs them in
topological order while sharing a mutable context dictionary.
"""

import contextlib
import importlib
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

import networkx as nx
import torch

from hmlib.aspen.plugins.base import Plugin
from hmlib.log import get_logger
from hmlib.utils.containers import SidebandQueue as Queue
from hmlib.utils.containers import create_queue

logger = get_logger(__name__)


@dataclass
class _Node:
    name: str
    cls_path: str
    depends: List[str]
    params: Dict[str, Any]
    module: torch.nn.Module
    stream: torch.cuda.Stream = None  # type: ignore[assignment]


class AspenNet(torch.nn.Module):
    """Configurable directed-acyclic graph runner for Aspen plugins.

    - Loads a YAML-like dict (already parsed) with node definitions.
    - Each node has: name, class (import path), depends (list), and params.
    - Executes nodes in topological order, passing and accumulating a
      shared context dict across nodes.

    @see @ref hmlib.aspen.plugins.base.Plugin "Plugin" for the plugin interface.
    """

    def __init__(
        self,
        name: str,
        graph_cfg: Dict[str, Any],
        shared: Optional[Dict[str, Any]] = None,
        minimal_context: bool = False,
        max_concurrent: int = 3,
        verbose: bool = False,
        validate_output_keys_each_time: bool = False,
    ):
        super().__init__()
        self.name: str = self._normalize_name(name)
        self._safe_name: str = self._sanitize_name(self.name)
        self.dot_path: str = os.path.abspath(f"aspennet_{self._safe_name}.dot")
        self._last_dot_path: Optional[str] = None
        self._verbose = verbose
        self.shared: Dict[str, Any] = shared or {}
        self.nodes: List[_Node] = []
        self.max_concurrent: int = max_concurrent
        self.num_concurrent: int = 0
        self._thread_error: Optional[BaseException] = None
        # Track which plugins have already had their output_keys() contract validated.
        self._output_keys_validated: Set[str] = set()
        # NetworkX DiGraph storing the plugins graph and attributes
        self.graph: nx.DiGraph = nx.DiGraph()
        self.minimal_context = bool(
            minimal_context
            or (isinstance(graph_cfg, dict) and graph_cfg.get("minimal_context", False))
        )
        pipeline_cfg: Dict[str, Any] = {}
        if isinstance(graph_cfg, dict):
            pipeline_cfg = graph_cfg.get("pipeline", {}) or {}
        if not isinstance(pipeline_cfg, dict):
            raise ValueError("AspenNet 'pipeline' configuration must be a mapping if provided.")
        threaded_flag = pipeline_cfg.get("threaded")
        if threaded_flag is None and isinstance(graph_cfg, dict):
            threaded_flag = graph_cfg.get("threaded_trunks", False)
        self.threaded_trunks: bool = bool(threaded_flag)
        output_check_flag = pipeline_cfg.get("check_output_keys_each_time", None)
        if output_check_flag is None and isinstance(graph_cfg, dict):
            output_check_flag = graph_cfg.get("check_output_keys_each_time", None)
        self.check_output_keys_each_time: bool = bool(
            validate_output_keys_each_time or output_check_flag
        )
        queue_size_cfg = pipeline_cfg.get("queue_size", 1)
        try:
            self.thread_queue_size: int = max(1, int(queue_size_cfg))
        except Exception as exc:
            raise ValueError(
                f"AspenNet pipeline queue_size must be an integer, got {queue_size_cfg!r}"
            ) from exc
        cuda_streams_flag = pipeline_cfg.get("cuda_streams", True)
        self.thread_cuda_streams: bool = bool(cuda_streams_flag)

        # Accept a dict with a required {plugins: {...}} mapping.
        plugins = graph_cfg.get("plugins") if isinstance(graph_cfg, dict) else None
        if plugins is None:
            raise ValueError("AspenNet expects a dict with a 'plugins' mapping.")

        # Profiler wiring (optional and zero-overhead when absent)
        self._profiler = self.shared.get("profiler", None)

        self._build_nodes(plugins)
        self._build_graph(plugins)
        self.exec_order = self._toposort()
        self.training: bool = False
        self._iter_num: int = 0
        self.save_graphviz(self.dot_path)
        self.initialized = False

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for node in self.nodes:
            node.module.to(*args, **kwargs)
        return self

    @staticmethod
    def _normalize_name(name: str) -> str:
        if name is None:
            raise ValueError("AspenNet requires a non-empty name.")
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("AspenNet requires a non-empty name.")
        return normalized

    @staticmethod
    def _sanitize_name(name: str) -> str:
        safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("_")
        safe = safe.lstrip(".")
        return safe or "aspen"

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
    def _build_nodes(self, plugins: Dict[str, Any]):
        for name, spec in plugins.items():
            if spec is None:
                raise ValueError(f"Empty spec for plugin '{name}'")
            cls_path = spec.get("class")
            if not cls_path:
                raise ValueError(f"Plugin '{name}' missing 'class'")
            depends = list(spec.get("depends", []) or [])
            params = spec.get("params", {}) or {}
            enabled = spec.get("enabled", True)
            if not enabled:
                # Create a no-op stub to keep graph shape predictable
                module = _NoOpPlugin(name=name)
            else:
                module = self._instantiate(cls_path, params)
            if isinstance(module, Plugin):
                module.set_profiler(self._profiler)
            node = _Node(
                name=name, cls_path=cls_path, depends=depends, params=params, module=module
            )
            setattr(self, f"trunk_{name}", module)
            self.nodes.append(node)

    def _build_graph(self, plugins: Dict[str, Any]):
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
            raise ValueError(f"Unknown dependencies referenced in plugins: {details}")

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
            raise ValueError(f"Cycle detected in plugins graph: {cycle_nodes}")

        name2node: Dict[str, _Node] = {n.name: n for n in self.nodes}
        order_names: List[str] = list(nx.topological_sort(self.graph))
        return [name2node[n] for n in order_names]

    # endregion

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all plugins in topological order.

        - 'context' is a mutable dict; plugins can read and write.
        - Returns the final context for convenience.
        """
        # Ensure plugins can access shared resources
        context.setdefault("shared", self.shared)
        context.setdefault("plugins", {})
        if self.threaded_trunks:
            self._maybe_reraise_thread_error()
            return self._forward_threaded(context)
        grad_ctx = torch.enable_grad() if self.training else torch.no_grad()
        with grad_ctx:
            do_trace: bool = True and self._iter_num == 10
            if do_trace:
                pass
            for node in self.exec_order:
                self._execute_node(node, context)
            if do_trace:
                pass
        self._iter_num += 1
        return context

    # region graph export/visualization
    def to_networkx(self) -> nx.DiGraph:
        """Return a shallow copy of the internal NetworkX DiGraph."""
        return self.graph.copy()

    def _dot_lines(self) -> Iterable[str]:
        # Simple DOT writer without extra deps
        yield "digraph AspenNet {"
        yield "  rankdir=TB;"
        yield "  node [shape=box, style=rounded];"
        graph_label = self.name.replace("\\", "\\\\").replace('"', '\\"')
        yield f'  label="{graph_label}";'
        yield "  labelloc=t;"
        # Nodes with labels
        for n, data in self.graph.nodes(data=True):
            node_label = f"{n}\n{data.get('cls_path', '')}"
            yield f'  "{n}" [label="{node_label}"];'
        # Edges
        for u, v in self.graph.edges():
            yield f'  "{u}" -> "{v}";'
        yield "}"

    def to_dot(self) -> str:
        """Return the Graphviz DOT string for the plugins graph."""
        return "\n".join(self._dot_lines())

    def save_graphviz(self, path: str) -> None:
        """
        Save the plugins graph as a Graphviz DOT file.

        Args:
            path: Destination file path (e.g., "graph.dot").
        """
        dot = self.to_dot()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(dot)
        self._last_dot_path = os.path.abspath(path)

    def display_graphviz(self) -> None:
        """
        Display the plugins graph.

        Tries, in order:
        - xdot executable (if available) for an interactive popup
        - graphviz.Source (if `graphviz` python package is installed)
        - matplotlib via networkx (if matplotlib is available)
        - Prints DOT to stdout as a fallback
        """
        dot = self.to_dot()
        dot_path = self._last_dot_path or self.dot_path

        # Try xdot binary
        try:
            xdot_bin = shutil.which("xdot")
            if xdot_bin:
                path = dot_path or os.path.abspath("aspennet.dot")
                self.save_graphviz(path)
                subprocess.Popen([xdot_bin, path])
                return
        except Exception as ex:
            print(f"AspenNet: xdot display failed: {ex}")

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
            nx.draw(
                self.graph,
                pos,
                with_labels=True,
                node_size=1500,
                node_color="#DDEEFF",
                font_size=8,
                arrows=True,
            )
            plt.title("AspenNet Plugins Graph")
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

    def finalize(self) -> None:
        """Invoke ``finalize`` on all plugins if they provide it."""
        for node in self.nodes:
            finalize_fn = getattr(node.module, "finalize", None)
            if callable(finalize_fn):
                try:
                    finalize_fn()
                except Exception:
                    logger.exception("Aspen plugin %s finalize failed", node.name)

    # endregion

    def _maybe_reraise_thread_error(self) -> None:
        """Raise any exception captured from threaded plugin workers."""
        err = self._thread_error
        if err is None:
            return
        # Prefer wrapped exceptions that preserve original traceback.
        reraise = getattr(err, "reraise", None)
        if callable(reraise):
            reraise()
        raise err

    def _execute_node(self, node: _Node, context: Dict[str, Any]) -> None:
        plugin = node.module
        subctx = self._make_subcontext(plugin, context) if self.minimal_context else context
        name = f"aspen.plugin.{node.name}"
        if self._verbose:
            print(f"AspenNet: Executing plugin '{node.name}' with class '{node.cls_path}'")
        if isinstance(plugin, Plugin):
            prof_ctx = plugin.profile_scope(name)
        elif getattr(self._profiler, "enabled", False):
            prof_ctx = self._profiler.rf(name)
        else:
            prof_ctx = contextlib.nullcontext()
        with prof_ctx:
            out = plugin(subctx) or {}

        declared = set(getattr(plugin, "output_keys", lambda: set())())
        returned_keys = set(out.keys())

        if declared:
            should_check = self.check_output_keys_each_time or (
                node.name not in self._output_keys_validated
            )
            if should_check:
                extra_keys = returned_keys - declared
                if extra_keys:
                    raise ValueError(
                        f"AspenNet plugin '{node.name}' ({node.cls_path}) returned keys "
                        f"{sorted(extra_keys)} not declared in output_keys(). "
                        f"Declared keys: {sorted(declared)}"
                    )
                if not self.check_output_keys_each_time:
                    self._output_keys_validated.add(node.name)

        update_keys = declared if declared else returned_keys

        from .plugins.base import DeleteKey  # local import avoids cycle

        for key in update_keys:
            if key in out:
                value = out[key]
                if isinstance(value, DeleteKey):
                    if key in context:
                        del context[key]
                else:
                    context[key] = value

        context["plugins"][node.name] = {k: out[k] for k in out.keys()}

    def _make_subcontext(self, plugin: torch.nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        req_keys = set(getattr(plugin, "input_keys", lambda: set())())
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
        if not self.initialized:
            self.initialized = True
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
                in_queue = self.queues[index]
                is_last = index == len(self.exec_order) - 1
                out_queue = self.queues[index + 1]
                try:
                    while True:
                        item = in_queue.get()
                        if is_last:
                            pass
                        if item is stop_token:
                            out_queue.put(stop_token)
                            break
                        if isinstance(item, _ExceptionWrapper):
                            out_queue.put(item)
                            self.stop(wait=False)
                            break
                        grad_ctx = make_grad_ctx()
                        try:
                            with grad_ctx:
                                self._execute_with_stream(node, item)
                            if not is_last:
                                out_queue.put(item)
                            else:
                                assert self.num_concurrent > 0
                                self.num_concurrent -= 1
                        except BaseException as exc:
                            wrapper = _ExceptionWrapper(exc)
                            self._thread_error = wrapper
                            print(exc)
                            # Propagate the wrapped exception downstream when possible
                            if not is_last:
                                out_queue.put(wrapper)
                            else:
                                if self.num_concurrent > 0:
                                    self.num_concurrent -= 1
                            break
                finally:
                    print(f"AspenNet: Thread for plugin '{node.name}' exiting.")

            self.queues: List[Queue] = [
                create_queue(
                    mp=False,
                    name=f"Aspen-{self.exec_order[i-1].name}",
                    max_size=self.thread_queue_size,
                )
                for i in range(len(self.exec_order) + 1)
            ]
            self.threads = []
            for idx, node in enumerate(self.exec_order):
                thread = threading.Thread(
                    target=worker, args=(idx, node), daemon=True, name=node.name
                )
                thread.start()
                self.threads.append(thread)
        while self.num_concurrent >= self.max_concurrent:
            time.sleep(0.01)
        self.queues[0].put(context)
        self.num_concurrent += 1
        return None

    def stop(self, wait: bool = True) -> None:
        """Stop all threaded plugins and join their threads."""
        if not self.threaded_trunks or not hasattr(self, "queues"):
            return
        stop_token = object()
        for _ in self.exec_order:
            self.queues[0].put(stop_token)
        for thread in self.threads:
            if wait and thread.is_alive():
                thread.join()
        for q in self.queues:
            q.close()
        del self.queues

    def _execute_with_stream(self, node: _Node, context: Dict[str, Any]) -> None:
        use_cuda_stream = (
            self.thread_cuda_streams  # and torch.cuda.is_available() and device is not None
        )
        if use_cuda_stream:
            if node.stream is None:
                device = self._infer_device(context)
                node.stream = torch.cuda.Stream(
                    device=device,
                )
            # Ensure plugins that fetch context["cuda_stream"] see the stream actually running them.
            prev_stream = context.get("cuda_stream")
            has_prev_stream = "cuda_stream" in context
            context["cuda_stream"] = node.stream
            try:
                with torch.cuda.stream(node.stream):
                    self._execute_node(node, context)
                # Ensure all work enqueued on this trunk's stream has
                # completed before handing the context to downstream
                # plugins. This preserves per-frame ordering while still
                # allowing different trunks to overlap on separate streams.
                node.stream.synchronize()
            finally:
                if has_prev_stream:
                    context["cuda_stream"] = prev_stream
                else:
                    context.pop("cuda_stream", None)
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


class _NoOpPlugin(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self._name = name

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        # Intentionally does nothing
        return {}
