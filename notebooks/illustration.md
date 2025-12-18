---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

This notebook draws a simple figure to illustrate the redirection graph.

```python
import colorsys
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


mpl.style.use("prl.mplstyle")
```

```python
def scale_lightness(rgb, scale_l):
    # https://stackoverflow.com/a/60562502/1150961
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)
```

```python
def round_hull(xy, radius, num=50):
    xy = np.asarray(xy)
    radius = radius * np.ones(xy.shape[0])
    angle = np.linspace(0, 2 * np.pi, num)
    delta = radius[:, None, None] * np.transpose([np.cos(angle), np.sin(angle)])
    xy = xy[:, None] + delta
    return ConvexHull(xy.reshape((-1, 2)))


def round_hull_polygon(xy, radius, num=50, **kwargs):
    hull = round_hull(xy, radius, num)
    return mpl.patches.Polygon(hull.points[hull.vertices], **kwargs)


radius = 0.2
linewidth = 1
node_kwargs = {
    "facecolor": "white",
    "edgecolor": "black",
    "linewidth": linewidth,
}
nodes = {
    0: {
        "xy": (0, 0),
        "edgecolor": "C0",
        "facecolor": scale_lightness(mpl.colors.to_rgb("C0"), 2),
        "wedge": None,
        # "label": "$\\sigma_t$",
        "label": 7,
    },
    1: {"xy": (0, 1), "label": "4"},
    2: {"xy": (0, 2), "label": "1"},
    3: {
        "xy": (1, 0),
        "edgecolor": "C1",
        "facecolor": scale_lightness(mpl.colors.to_rgb("C1"), 1.5),
        "wedge": None,
        # "label": "$s$",
        "label": 6,
    },
    4: {"xy": (1, 1), "label": "5"},
    5: {"xy": (1, 2), "label": "2"},
    6: {"xy": (2, 2), "label": "3"},
}

edges = [
    (3, 4),
    (3, 1),
    (1, 2),
    (2, 5),
    (5, 6),
    # (1, 4),
]

fig, ax = plt.subplots()
ax.set_aspect("equal")

# Plot edges.
segments = [[nodes[i]["xy"], nodes[j]["xy"]] for (i, j) in edges]
lines = mpl.collections.LineCollection(segments, color="black", zorder=1, linewidth=linewidth)
ax.add_collection(lines)

# Plot nodes.
node_patches = {}
for label, kwargs in nodes.items():
    kwargs = node_kwargs | kwargs
    wedge = kwargs.pop("wedge", None)
    node = mpl.patches.Circle(**kwargs, radius=radius)
    ax.add_patch(node)
    node_patches[label] = node

    text = kwargs.get("label")
    if text is not None:
        color = "k"
        ax.text(*kwargs["xy"], str(text), ha="center", va="center", color=color)

    if wedge == "left":
        patch = mpl.patches.Wedge(kwargs["xy"], radius, 90, 270, facecolor=node.get_edgecolor())
    elif wedge == "top":
        patch = mpl.patches.Wedge(kwargs["xy"], radius, 0, 180, facecolor=node.get_edgecolor())
    elif wedge == "square":
        x, y = kwargs["xy"]
        width = radius
        patch = mpl.patches.Rectangle((x - width / 2, y - width / 2), width, width, facecolor=node.get_edgecolor())
    else:
        continue
    ax.add_patch(patch)

# Plot the new edge.
arrowstyle = "-|>, head_length=3, head_width=1.5"
new_edge = mpl.patches.FancyArrowPatch(
    posA=nodes[0]["xy"],
    posB=nodes[1]["xy"],
    patchA=node_patches[0],
    patchB=node_patches[1],
    color=nodes[0]["edgecolor"],
    linewidth=linewidth,
    arrowstyle=arrowstyle,
)
ax.add_patch(new_edge)

# Plot the hop.
hop = mpl.patches.FancyArrowPatch(
    posA=nodes[3]["xy"],
    posB=nodes[1]["xy"],
    patchA=node_patches[3],
    patchB=node_patches[1],
    color=nodes[3]["edgecolor"],
    linewidth=linewidth,
    arrowstyle=arrowstyle,
    connectionstyle="arc3,rad=0.3",
)
ax.add_patch(hop)

wrappers = [
    ([1, 3, 4], radius + 0.1, {"ls": "--"}),
    ([1, 3, 4, 2, 5, 0], radius + np.asarray([0.15, 0.15, 0.15, 0.1, 0.1, 0.1]), {"ls": ":"}),
]
for idx, rad, kwargs in wrappers:
    poly = round_hull_polygon(
        [nodes[i]["xy"] for i in idx], rad, zorder=0, edgecolor="gray",
        facecolor=mpl.colors.to_rgba("gray", 0.1), **kwargs,
    )
    ax.add_patch(poly)

ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 2.5)
ax.set_axis_off()


handle_kwargs = {"marker": "o", "ls": "none", "markersize": 7}
old_handle = mpl.lines.Line2D([], [], markerfacecolor="none", markeredgecolor="black", **handle_kwargs)
new_handle = mpl.lines.Line2D([], [], fillstyle="full", color="C0", **handle_kwargs)
seed_handle = mpl.lines.Line2D([], [], fillstyle="full", color="C1", **handle_kwargs)
new_edge_handle = mpl.lines.Line2D([], [], color="C0", **(handle_kwargs | {"marker": r"$\rightarrow$"}))
hop_handle = mpl.lines.Line2D([], [], color="C1", **(handle_kwargs | {"marker": r"$\rightarrow$"}))
actual_handle = mpl.lines.Line2D([], [], color="gray", ls="--")
possible_handle = mpl.lines.Line2D([], [], color="gray", ls=":")
handles_labels = [
    (new_handle, r"source node $\sigma_t$"),
    (seed_handle, r"seed node $s\in S_{t}$"),
    (old_handle, r"other nodes"),
    (hop_handle, "redirection with\nprobability $\\theta$"),
    (new_edge_handle, r"new edge $\epsilon_{t}$"),
    (actual_handle, r"subgraph $B^{(k)}_{s}$"),
    (possible_handle, r"receptive field $B^{(r)}_{\sigma_t}$"),
]
ax.legend(*zip(*handles_labels), loc="lower right", frameon=False)

fig.tight_layout()
fig.get_size_inches()
fig.savefig("redirection-illustration.pdf", bbox_inches="tight")
```
