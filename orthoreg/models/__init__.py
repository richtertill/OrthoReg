"""Hybrid model implementations combining symbolic and neural components.

Submodules are imported on demand to avoid a circular import with
``orthoreg.setup`` (which depends on ``orthoreg.models.networks`` /
``forecasters`` while ``orthoreg.models.exp`` depends back on
``orthoreg.setup``).
"""
