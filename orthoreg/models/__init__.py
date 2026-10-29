"""Hybrid model implementations combining symbolic and neural components.

Submodules are imported on demand to avoid a circular import with
``orthoreg.feature_library`` (which depends on ``orthoreg.models.networks`` /
``forecasters`` while ``orthoreg.models.experiment`` depends back on
``orthoreg.feature_library``).
"""
