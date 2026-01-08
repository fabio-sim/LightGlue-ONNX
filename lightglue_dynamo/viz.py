"""
2D visualization primitives based on Matplotlib.
1) Plot images with `plot_images`.
2) Call `plot_keypoints` or `plot_matches` any number of times.
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.cm
import matplotlib.patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

Color = str | tuple[float, float, float] | tuple[float, float, float, float]


def plot_images(
    imgs: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    cmaps: str | Sequence[str] = "gray",
    dpi: int = 100,
    pad: float = 0.5,
    adaptive: bool = True,
) -> None:
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, BGR (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    cmaps_list = [str(cmap) for cmap in (list(cmaps) if isinstance(cmaps, list | tuple) else [cmaps] * n)]

    ratios = [i.shape[1] / i.shape[0] for i in imgs] if adaptive else [4 / 3] * n
    figsize = [sum(ratios) * 4.5, 4.5]
    ax: list
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios})  # type: ignore
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i][..., ::-1], cmap=plt.get_cmap(cmaps_list[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_keypoints(
    kpts: Sequence[np.ndarray],
    colors: Color | Sequence[Color] = "lime",
    ps: float = 4,
    axes: Sequence[Axes] | None = None,
    a: float | Sequence[float] = 1.0,
) -> None:
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    colors_list = list(colors) if isinstance(colors, list | tuple) else [colors] * len(kpts)
    alpha_list = [float(value) for value in a] if isinstance(a, list | tuple) else [cast(float, a)] * len(kpts)
    axes_list = plt.gcf().axes if axes is None else axes
    for ax, k, c, alpha in zip(axes_list, kpts, colors_list, alpha_list, strict=False):
        ax.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0, alpha=alpha)


def plot_matches(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    color: Color | Sequence[Color] | None = None,
    lw: float = 1.5,
    ps: float = 4,
    a: float = 1.0,
    labels: Sequence[str] | None = None,
    axes: Sequence[Axes] | None = None,
) -> None:
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    if axes is None:
        ax = fig.axes
        ax0, ax1 = ax[0], ax[1]
    else:
        ax0, ax1 = axes

    assert len(kpts0) == len(kpts1)
    if color is None:
        color_list = [tuple(c) for c in matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()]  # type: ignore
    elif isinstance(color, list | tuple):
        if len(color) == 0:
            color_list = []
        elif isinstance(color[0], tuple | list):
            color_list = [tuple(c) if isinstance(c, list) else c for c in color]
        else:
            color_list = list(color)
    else:
        color_list = [color] * len(kpts0)

    if lw > 0:
        for i in range(len(kpts0)):
            line = matplotlib.patches.ConnectionPatch(
                xyA=(kpts0[i, 0], kpts0[i, 1]),
                xyB=(kpts1[i, 0], kpts1[i, 1]),
                coordsA=ax0.transData,
                coordsB=ax1.transData,
                axesA=ax0,
                axesB=ax1,
                zorder=1,
                color=color_list[i],
                linewidth=lw,
                clip_on=True,
                alpha=a,
                label=None if labels is None else labels[i],
                picker=5.0,
            )
            line.set_annotation_clip(True)
            fig.add_artist(line)

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color_list, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color_list, s=ps)


def add_text(
    idx: int,
    text: str,
    pos: tuple[float, float] = (0.01, 0.99),
    fs: int = 15,
    color: str = "w",
    lcolor: str | None = "k",
    lwidth: float = 2,
    ha: str = "left",
    va: str = "top",
) -> None:
    ax = plt.gcf().axes[idx]
    t = ax.text(*pos, text, fontsize=fs, ha=ha, va=va, color=color, transform=ax.transAxes)
    if lcolor is not None:
        t.set_path_effects([path_effects.Stroke(linewidth=lwidth, foreground=lcolor), path_effects.Normal()])


def save_plot(path: str | Path, **kw: object) -> None:
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)
