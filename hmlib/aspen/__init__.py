"""Aspen graph-based inference engine and trunk interfaces.

Aspen treats the end-to-end pipeline as a directed-acyclic graph of "plugins"
that process and annotate a shared context dictionary.

@see @ref hmlib.aspen.net.AspenNet "AspenNet" for the graph executor.
@see @ref hmlib.aspen.plugins.base.Plugin "Plugin" for the trunk base class.
"""

from .net import AspenNet
from .plugins.base import Plugin

__all__ = ["AspenNet", "Plugin"]
