from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from hmlib.builder import HM

from .base import Plugin


@HM.register_module()
class JoinPlugin(Plugin):
    """Explicit join node for parallel AspenNet branches.

    This plugin is the *only* supported exception to AspenNet's "unique inheritance path"
    validation. It serves as a semantic barrier: upstream ancestry is intentionally
    collapsed at this node so downstream plugins cannot receive duplicated ancestry
    via multiple upstream paths.

    By default this plugin does not modify the context; it only validates that the
    configured upstream plugins have executed for the current context and emits a
    lightweight join token.
    """

    # Marker used by AspenNet validation to allow merges at this node.
    allow_multi_path_inputs = True

    def __init__(
        self,
        enabled: bool = True,
        *,
        required_plugins: Optional[List[str]] = None,
        output_key: str = "join_token",
    ) -> None:
        super().__init__(enabled=enabled)
        self._required_plugins = [str(p) for p in (required_plugins or []) if p]
        self._output_key = str(output_key or "join_token")

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        seq = context.get("_aspen_seq")
        shared = context.get("shared", {})
        if not isinstance(shared, dict):
            shared = {}
            context["shared"] = shared

        plugins = context.get("plugins")
        if not isinstance(plugins, dict):
            raise ValueError(
                "JoinPlugin requires context['plugins'] to be a dict; "
                "AspenNet should have created this automatically."
            )

        missing = [name for name in self._required_plugins if name not in plugins]
        if missing:
            available = sorted(str(k) for k in plugins.keys())
            raise ValueError(
                "JoinPlugin missing required upstream outputs. "
                f"Missing: {missing}. Available plugins: {available}."
            )

        join_state = shared.setdefault("_aspen_join_state", {})
        join_key = str(id(self))
        if seq is not None:
            last_seq = join_state.get(join_key)
            if last_seq == seq:
                raise ValueError(
                    "JoinPlugin received the same batch more than once before forwarding. "
                    f"seq={seq!r} required_plugins={self._required_plugins}"
                )
            join_state[join_key] = seq

        merged: Dict[str, Any] = {}
        provenance: Dict[str, str] = {}
        for plugin_name in self._required_plugins:
            plugin_out = plugins.get(plugin_name)
            if plugin_out is None:
                raise ValueError(
                    "JoinPlugin missing required upstream plugin output unexpectedly: "
                    f"{plugin_name!r}"
                )
            if not isinstance(plugin_out, dict):
                raise ValueError(
                    "JoinPlugin expected context['plugins'][name] to be a dict. "
                    f"name={plugin_name!r} got={type(plugin_out).__name__}"
                )
            for key, value in plugin_out.items():
                if key in merged:
                    raise ValueError(
                        "JoinPlugin detected duplicate keys across inputs. "
                        f"key={key!r} first_from={provenance[key]!r} second_from={plugin_name!r} "
                        f"required_plugins={self._required_plugins}"
                    )
                merged[key] = value
                provenance[key] = plugin_name

        return {self._output_key: merged}

    def input_keys(self) -> Set[str]:
        return {"plugins"}

    def output_keys(self) -> Set[str]:
        return {self._output_key}
