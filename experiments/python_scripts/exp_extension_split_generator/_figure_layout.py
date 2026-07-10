"""Shared figure-layout utilities for the ansatz-form experiment plots.

Used by both the per-run comparison plots (``_ansatz_forms_plots.py``) and the
cross-seed summary (``ansatz_forms_cross_seed_summary.py``). Provides:

* :func:`formula_box` -- a LaTeX text block below a figure stating the plotted
  quantity;
* :func:`check_layout` -- non-fatal warnings for common layout defects (a
  legend / title / axis label spilling past the figure canvas; the bottom
  formula box overlapping the x-axis tick labels / xlabel, i.e. hiding them);
* :func:`finalize_figure` -- add the formula box, run the layout check, and save
  with ``bbox_extra_artists`` so external legends are never actually clipped.
"""
from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt


def formula_box(fig, text, fontsize=8):
    """Add a LaTeX text block below the figure; return the artist (or None)."""
    if not text:
        return None
    return fig.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=fontsize,
                    bbox=dict(boxstyle="round", facecolor="#f5f5f5",
                              edgecolor="#bbbbbb"))


def _bb(artist, renderer):
    """Window extent of an artist, or None if it has no drawable extent."""
    try:
        bb = artist.get_window_extent(renderer)
    except Exception:
        return None
    return bb if (bb is not None and bb.width > 0 and bb.height > 0) else None


def _overlaps(a, b, eps=1.0):
    return not (a.x1 < b.x0 + eps or b.x1 < a.x0 + eps
                or a.y1 < b.y0 + eps or b.y1 < a.y0 + eps)


def check_layout(fig, fname, *, legends=(), formula=None, axes=()):
    """Emit warnings for common figure-layout defects.

    1. **Cropping risk** -- any legend / title / x- or y-axis label whose box
       spills past the figure canvas (rescued by ``bbox_extra_artists`` on save,
       but the spill means the layout under-reserves space).
    2. **Hidden x-axis** -- the bottom formula box overlapping the x-axis tick
       labels or the x-axis label, which would hide them.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    eps = 2.0

    named = [("legend", lg) for lg in legends if lg is not None]
    if formula is not None:
        named.append(("formula box", formula))
    for ax in axes:
        named += [("title", ax.title), ("x-axis label", ax.xaxis.label),
                  ("y-axis label", ax.yaxis.label)]
    for kind, art in named:
        bb = _bb(art, r)
        if bb is None:
            continue
        if (bb.x0 < fb.x0 - eps or bb.x1 > fb.x1 + eps
                or bb.y0 < fb.y0 - eps or bb.y1 > fb.y1 + eps):
            warnings.warn(f"[{fname}] {kind} spills past the figure area "
                          f"(cropping risk) — reserve more margin/legend space.",
                          stacklevel=2)

    if formula is not None:
        fbb = _bb(formula, r)
        for ax in axes:
            for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
                tb = _bb(t, r)
                if fbb is not None and tb is not None and _overlaps(fbb, tb):
                    warnings.warn(f"[{fname}] the formula box overlaps the x-axis "
                                  f"labels (may hide them) — increase the reserved "
                                  f"bottom margin.", stacklevel=2)
                    break
        # 3. **Formula hidden by a legend** — a tall (multi-line) formula box can
        #    rise into a bottom-anchored legend and disappear behind it.
        for lg in legends:
            lb = _bb(lg, r)
            if fbb is not None and lb is not None and _overlaps(fbb, lb):
                warnings.warn(f"[{fname}] the formula box overlaps a legend (may "
                              f"hide the formula) — raise the legend anchor or "
                              f"reserve more bottom margin.", stacklevel=2)
                break


def finalize_figure(fig, path, *, legends=(), formula=None, axes=(), dpi=140,
                    formula_fontsize=8):
    """Add the formula box, check the layout, and save including all legends."""
    fname = os.path.basename(str(path))
    legs = [lg for lg in legends if lg is not None]
    formula_art = formula_box(fig, formula, fontsize=formula_fontsize) if formula else None
    extra = legs + ([formula_art] if formula_art is not None else [])
    check_layout(fig, fname, legends=legs, formula=formula_art, axes=axes)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", bbox_extra_artists=extra)
    plt.close(fig)
