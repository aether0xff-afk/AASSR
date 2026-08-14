"""Environment I/O plugins for AASSR.

New plugins must follow :mod:`aassr_v2.core.plugin_contract`: command syntax,
public observation data types, real I/O, and external lifecycle/reward passthrough
only. Learning representations and task semantics belong to the Core.

``current_pentest`` is retained only for historical 10k checkpoint/reproduction
compatibility and intentionally does not define the new plugin-authoring model.
"""
