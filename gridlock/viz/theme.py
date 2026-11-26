"""Look and feel for the viewers: palette, fonts, panels and chart styling.

The style is brutalist, after part.archi: white paper, black ink, 1px rules instead of cards and
shadows, square corners, lowercase numbered labels, regular weights only (no bold), and a single
signal red kept for monitor flags. Inverse (black) panels hold reference text.

Everything is drawn with plain matplotlib. UI text uses the first installed font of MONO, display
text (titles, agent names) the first of SANS; DejaVu Sans, bundled with matplotlib, fills in
symbols such as ⚑ ★ ⚱.
"""

from contextlib import contextmanager
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- palette -----------------------------------------------------------------------------------
PAPER = "#ffffff"
INK = "#000000"           # text, rules, walls, playhead
SIGNAL = "#f24a3c"        # the one colour: monitor flags only (no agent or tile uses red)
GRAY = "#a5a5a5"          # flat secondary fill
YELLOW = "#efc14e"        # loopholes, on inverse panels only

BG = PAPER                # page
CARD = PAPER
CARD_EDGE = INK           # panel borders are rules, not soft edges
INVERSE = INK             # inverse panel background
INVERSE_INK = PAPER
INVERSE_MUTED = "#9a9a9a"
MUTED = "#6b6b6b"         # secondary text
FAINT = GRAY              # tertiary text
GRIDLINE = "#e4e4e4"
ACCENT = INK              # playhead, progress, primary controls
ACCENT_DARK = SIGNAL      # hover
FLAG = SIGNAL
LOOPHOLE = GRAY           # ground-truth loophole steps on time axes (hatched)
GOOD = INK
WARN = INK                # R* violations: black ✕, so red stays the monitor's

RULE_W = 1.0              # points
HAIR_W = 0.5

SANS = ("Helvetica Neue", "Helvetica", "Arial", "Liberation Sans")
MONO = ("Menlo", "SF Mono", "IBM Plex Mono", "Consolas", "Liberation Mono")


@lru_cache(maxsize=None)
def _installed(candidates: tuple[str, ...], fallback: str) -> list[str]:
    """The first installed candidate with a regular upright face, then the fallback, then DejaVu
    Sans for any symbols the others lack."""
    from matplotlib import font_manager
    names = {font.name for font in font_manager.fontManager.ttflist
             if font.style == "normal" and 300 <= int(font.weight) <= 500}
    return [name for name in candidates if name in names][:1] + list(dict.fromkeys([fallback, "DejaVu Sans"]))


def sans() -> list[str]:
    return _installed(SANS, "DejaVu Sans")


def mono() -> list[str]:
    return _installed(MONO, "DejaVu Sans Mono")


def rc() -> dict:
    return {
        "font.family": mono(), "font.sans-serif": sans(), "font.monospace": mono(), "font.size": 8.5, "font.weight": "normal", "text.color": INK,
        "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": PAPER,
        "axes.edgecolor": INK, "axes.linewidth": RULE_W, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.labelweight": "normal", "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "legend.fontsize": 7.5,
        "lines.solid_capstyle": "butt", "lines.dash_capstyle": "butt", "lines.solid_joinstyle": "miter",
        "hatch.linewidth": 0.6, "hatch.color": GRAY,
        "toolbar": "None",        # the viewers have their own controls; the toolbar only takes height
    }


@contextmanager
def themed():
    """Apply the theme to everything created inside the block (rcParams are read at creation)."""
    with plt.rc_context(rc()):
        yield


# --- layout helpers ----------------------------------------------------------------------------

class Layout:
    """Place axes and panels in inches from the figure's top-left corner."""

    def __init__(self, fig):
        self.fig = fig
        self.w, self.h = fig.get_size_inches()

    def rect(self, x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        return x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h

    def axes(self, x: float, y: float, w: float, h: float, **kwargs):
        return self.fig.add_axes(self.rect(x, y, w, h), **kwargs)

    def card(self, x: float, y: float, w: float, h: float, inverse: bool = False, lw: float = RULE_W) -> list:
        """A square panel outlined by a 1px rule (filled black when `inverse`), behind everything."""
        patch = Rectangle(self.rect(x, y, w, h)[:2], w / self.w, h / self.h, transform=self.fig.transFigure,
                          fc=INVERSE if inverse else CARD, ec=INK, lw=lw, zorder=-1)
        self.fig.patches.append(patch)
        return [patch]

    def rule(self, x: float, y: float, w: float, color: str = INK, lw: float = RULE_W):
        """A horizontal hairline from (x, y), w inches long."""
        line = plt.Line2D([x / self.w, (x + w) / self.w], [1 - y / self.h] * 2, transform=self.fig.transFigure,
                          color=color, lw=lw, solid_capstyle="butt", zorder=0)
        self.fig.lines.append(line)
        return line

    def text(self, x: float, y: float, s: str, **kwargs):
        """Figure text at (x, y) inches from the top-left; y is the text's top unless va says otherwise."""
        kwargs.setdefault("va", "top")
        return self.fig.text(x / self.w, 1 - y / self.h, s, **kwargs)

    def title(self, x: float, y: float, s: str, size: float = 34, **kwargs):
        """Display text: large, regular weight, set tight to the top."""
        kwargs.setdefault("color", INK)
        return self.text(x, y, s, fontsize=size, family=sans(), fontweight="normal", **kwargs)

    def edge_label(self, s: str, x: float = None):
        """Small text rotated up the right-hand margin, like a running head."""
        x = self.w - 0.11 if x is None else x
        return self.text(x, self.h / 2, s, rotation=90, ha="center", va="center", fontsize=6.5, color=MUTED)

    def bar(self, x: float, y: float, w: float, h: float, color: str, **kwargs):
        patch = Rectangle(self.rect(x, y, w, h)[:2], w / self.w, h / self.h, transform=self.fig.transFigure,
                          fc=color, ec="none", **kwargs)
        self.fig.patches.append(patch)
        return patch


def section_label(ax, text: str, color: str = INK, y: float = 1.0, pad_pts: float = 6, index=None) -> None:
    """Small lowercase label just above an axes' top-left corner, optionally numbered ("01 / …")."""
    label = text.lower() if index is None else f"{index:02d} / {text.lower()}"
    ax.annotate(label, xy=(0, y), xycoords="axes fraction", xytext=(0, pad_pts), textcoords="offset points",
                fontsize=7.5, color=color, va="bottom", ha="left", family=mono())


def style_chart(ax) -> None:
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(RULE_W)
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=HAIR_W * 1.5, pad=3, labelsize=7.5, labelcolor=INK, color=INK)


def pill(ax_or_fig, x: float, y: float, text: str, fc: str, color: str = PAPER, **kwargs):
    """Square boxed label; x, y in the transform given by kwargs (default: axes or figure fraction)."""
    kwargs.setdefault("fontsize", 7.5)
    return ax_or_fig.text(x, y, text, color=color, bbox=box(fc), **kwargs)


def box(fc: str = PAPER, ec: str = INK) -> dict:
    """bbox style for boxed labels: square, ruled, no rounding."""
    return dict(boxstyle="square,pad=0.45", fc=fc, ec=ec, lw=RULE_W)


def progress(layout: Layout, x: float, y: float, w: float, h: float = 0.06):
    """A hairline rule with a solid black fill that grows along it; returns (track, fill)."""
    layout.rule(x, y + h, w)
    track = layout.bar(x, y, w, h, "none")
    fill = layout.bar(x, y, w, h, ACCENT)
    return track, fill


def set_progress(fill, track, fraction: float) -> None:
    fill.set_width(track.get_width() * max(0.0, min(1.0, fraction)))


def clear_artists(artists: list) -> None:
    for artist in artists:
        try:
            artist.remove()
        except (ValueError, NotImplementedError):
            pass
    artists.clear()


__all__ = ["ACCENT", "ACCENT_DARK", "BG", "CARD", "CARD_EDGE", "FAINT", "FLAG", "GOOD", "GRAY", "INK", "INVERSE",
           "INVERSE_INK", "INVERSE_MUTED", "LOOPHOLE", "Layout", "MUTED", "PAPER", "Rectangle", "SIGNAL", "WARN",
           "YELLOW", "box", "clear_artists", "mono", "pill", "progress", "rc", "sans", "section_label",
           "set_progress", "style_chart", "themed"]
