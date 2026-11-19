"""Look and feel for the viewers: palette, fonts, card backgrounds and chart styling.

Everything is drawn with plain matplotlib. Text uses the first installed font of SANS (then
DejaVu Sans, bundled with matplotlib, which also supplies symbols such as ⚑ ★ ⚱).
"""

from contextlib import contextmanager
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# --- palette -----------------------------------------------------------------------------------
BG = "#eef2f7"            # page
CARD = "#ffffff"
CARD_EDGE = "#e2e8f0"
SHADOW = "#0f172a"
INK = "#0f172a"           # main text
MUTED = "#64748b"         # secondary text
FAINT = "#94a3b8"         # tertiary text, axes
GRIDLINE = "#eef2f7"
ACCENT = "#2563eb"
ACCENT_DARK = "#1d4ed8"
FLAG = "#dc2626"          # monitor flags only: no agent or tile uses red
LOOPHOLE = "#cbd5e1"      # ground-truth loophole steps on time axes
GOOD = "#059669"
WARN = "#b45309"

SANS = ("Inter", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "Liberation Sans")
MONO = ("Menlo", "SF Mono", "Consolas", "Liberation Mono")


@lru_cache(maxsize=None)
def _installed(candidates: tuple[str, ...], fallback: str) -> list[str]:
    """The first candidate installed with both a regular and a bold face (else no bold text),
    then the fallback, which also fills in any missing symbols."""
    from matplotlib import font_manager
    weights: dict[str, set] = {}
    for font in font_manager.fontManager.ttflist:
        if font.style == "normal":
            weights.setdefault(font.name, set()).add(int(font.weight) >= 600)
    return [name for name in candidates if weights.get(name) == {False, True}][:1] + [fallback]


def sans() -> list[str]:
    return _installed(SANS, "DejaVu Sans")


def mono() -> list[str]:
    return _installed(MONO, "DejaVu Sans Mono")


def rc() -> dict:
    return {
        "font.family": sans(), "font.size": 9, "text.color": INK,
        "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": CARD,
        "axes.edgecolor": CARD_EDGE, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": FAINT, "ytick.color": FAINT, "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False, "legend.fontsize": 8,
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
        "toolbar": "None",        # the viewers have their own controls; the toolbar only takes height
    }


@contextmanager
def themed():
    """Apply the theme to everything created inside the block (rcParams are read at creation)."""
    with plt.rc_context(rc()):
        yield


# --- layout helpers ----------------------------------------------------------------------------

class Layout:
    """Place axes and cards in inches from the figure's top-left corner."""

    def __init__(self, fig):
        self.fig = fig
        self.w, self.h = fig.get_size_inches()

    def rect(self, x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        return x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h

    def axes(self, x: float, y: float, w: float, h: float, **kwargs):
        return self.fig.add_axes(self.rect(x, y, w, h), **kwargs)

    def card(self, x: float, y: float, w: float, h: float, radius: float = 0.12) -> list:
        """A white rounded card with a soft shadow, behind everything else."""
        fx, fy, fw, fh = self.rect(x, y, w, h)
        r = radius / self.w
        shadow = FancyBboxPatch((fx + 0.6 * r * 0.25, fy - 0.004), fw, fh, boxstyle=f"round,pad=0,rounding_size={r}",
                                transform=self.fig.transFigure, fc=SHADOW, ec="none", alpha=0.05, zorder=-2)
        card = FancyBboxPatch((fx, fy), fw, fh, boxstyle=f"round,pad=0,rounding_size={r}",
                              transform=self.fig.transFigure, fc=CARD, ec=CARD_EDGE, lw=0.8, zorder=-1)
        self.fig.patches.extend([shadow, card])
        return [shadow, card]

    def text(self, x: float, y: float, s: str, **kwargs):
        """Figure text at (x, y) inches from the top-left; y is the text's top unless va says otherwise."""
        kwargs.setdefault("va", "top")
        return self.fig.text(x / self.w, 1 - y / self.h, s, **kwargs)

    def bar(self, x: float, y: float, w: float, h: float, color: str, **kwargs):
        fx, fy, fw, fh = self.rect(x, y, w, h)
        patch = FancyBboxPatch((fx, fy), fw, fh, boxstyle=f"round,pad=0,rounding_size={h / 2 / self.h}",
                               transform=self.fig.transFigure, fc=color, ec="none", **kwargs)
        self.fig.patches.append(patch)
        return patch


def section_label(ax, text: str, color: str = MUTED, y: float = 1.0, pad_pts: float = 6) -> None:
    """Small bold caps label just above an axes' top-left corner."""
    ax.annotate(text.upper(), xy=(0, y), xycoords="axes fraction", xytext=(0, pad_pts), textcoords="offset points",
                fontsize=7.5, fontweight="bold", color=color, va="bottom", ha="left")


def style_chart(ax) -> None:
    ax.set_facecolor(CARD)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CARD_EDGE)
    ax.grid(axis="y", color=GRIDLINE, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=4, labelsize=8, labelcolor=MUTED)


def pill(ax_or_fig, x: float, y: float, text: str, fc: str, color: str = "white", **kwargs):
    """Rounded label; x, y in the transform given by kwargs (default: axes or figure fraction)."""
    kwargs.setdefault("fontsize", 8)
    kwargs.setdefault("fontweight", "bold")
    return ax_or_fig.text(x, y, text, color=color, bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8", fc=fc,
                                                              ec="none"), **kwargs)


def progress(layout: Layout, x: float, y: float, w: float, h: float = 0.06):
    """A thin progress track; returns (track, fill) so the fill's width can be updated."""
    track = layout.bar(x, y, w, h, CARD_EDGE)
    fill = layout.bar(x, y, w, h, ACCENT)
    return track, fill


def set_progress(fill, track, fraction: float) -> None:
    fill.set_width(max(track.get_height() * 0.0, track.get_width() * max(0.0, min(1.0, fraction))))


def clear_artists(artists: list) -> None:
    for artist in artists:
        try:
            artist.remove()
        except (ValueError, NotImplementedError):
            pass
    artists.clear()


__all__ = ["ACCENT", "ACCENT_DARK", "BG", "CARD", "CARD_EDGE", "FAINT", "FLAG", "GOOD", "INK", "LOOPHOLE",
           "Layout", "MUTED", "Rectangle", "WARN", "clear_artists", "mono", "pill", "progress", "rc", "sans",
           "section_label", "set_progress", "style_chart", "themed"]
