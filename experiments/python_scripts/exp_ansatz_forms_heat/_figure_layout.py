"""Compatibility shim: figure-layout helpers moved to the library.

The single source of truth is
:mod:`learning_option_pricing.utils.figure_layout`; this shim re-exports the
public names so the existing ``from _figure_layout import ...`` statements of
the experiment scripts keep working unchanged.
"""
from learning_option_pricing.utils.figure_layout import (  # noqa: F401
    check_layout,
    finalize_figure,
    formula_box,
)


__all__ = ["check_layout", "finalize_figure", "formula_box"]
