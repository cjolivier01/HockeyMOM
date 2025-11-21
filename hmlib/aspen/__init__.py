"""Aspen graph-based inference engine and trunk interfaces.

Aspen treats the end-to-end pipeline as a directed-acyclic graph of "trunks"
that process and annotate a shared context dictionary.

@see @ref hmlib.aspen.net.AspenNet "AspenNet" for the graph executor.
@see @ref hmlib.aspen.trunks.base.Trunk "Trunk" for the trunk base class.
"""

from .net import AspenNet
from .trunks.base import Trunk

__all__ = ["AspenNet", "Trunk"]
