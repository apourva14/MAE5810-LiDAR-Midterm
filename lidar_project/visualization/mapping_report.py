"""Compact report figure of saved 3D maps and trajectories.

Only retained map endpoints, executed states, and reference routes are read.
The report view does not construct maps, sense the scene, or plan trajectories.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import theme


def _load_selected(directory, case_ids):
    from ..mapping import load_map
    from ..trajectory.storage import load_result

    comparison = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in comparison["cases"]}
    selected = []
    for case_id in case_ids:
        if case_id not in by_id:
            raise ValueError(f"No saved environment named {case_id!r}")
        case = by_id[case_id]
        run = Path(case["run_directory"])
        if not run.is_absolute():
            run = directory / run
        selected.append((case, load_result(run), load_map(run)))
    extents = [
        np.asarray(mapping["metadata"]["extent_m"]) for _, _, mapping in selected
    ]
    if any(not np.allclose(extents[0], extent) for extent in extents[1:]):
        raise ValueError("The report comparison requires a common mapped region")
    return selected, extents[0]


def render_mapping_report(data_dir, output_dir):
    """Render the cluttered/narrow comparison at a 6.5 × 2.75 inch report size.

    Full final measured voxel maps are used. Low endpoints are omitted only from
    this display, using each saved map's ground_threshold_m. Reference and actual
    routes remain at z=0 because the simulated platform moves in a plane.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator

    directory, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    cases, extent = _load_selected(directory, ("cluttered", "narrow"))
    output.mkdir(parents=True, exist_ok=True)
    clouds = []
    for _, _, mapping in cases:
        xyz = np.asarray(mapping["map_xyz"])
        clouds.append(xyz[xyz[:, 2] > mapping["metadata"]["ground_threshold_m"]])
    maximum = max((float(xyz[:, 2].max()) for xyz in clouds if len(xyz)), default=1.0)
    upper = np.ceil(maximum * 2) / 2
    normalization = Normalize(0, upper)
    path_color = theme.color("figure_text")
    reference_color = theme.PALETTE[1]
    with plt.rc_context(theme.MATPLOTLIB):
        fig = plt.figure(figsize=(6.5, 2.75))
        fig.subplots_adjust(left=0.035, right=0.915, bottom=0.17, top=0.91, wspace=0.03)
        artist = None
        for i, ((spec, result, mapping), xyz) in enumerate(zip(cases, clouds)):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            artist = ax.scatter(
                *xyz.T,
                c=xyz[:, 2],
                cmap=theme.colormap("range"),
                norm=normalization,
                s=0.8,
                linewidths=0,
                depthshade=False,
                rasterized=True,
            )
            actual = np.asarray(result["states"]["xy"])
            reference = np.asarray(result["reference"]["xy"])
            ax.plot(
                reference[:, 0],
                reference[:, 1],
                np.zeros(len(reference)),
                color=reference_color,
                lw=1.05,
                ls=(0, (4, 2)),
            )
            ax.plot(
                actual[:, 0],
                actual[:, 1],
                np.zeros(len(actual)),
                color=path_color,
                lw=1.25,
            )
            ax.plot(*actual[0], 0, marker="o", ms=3, color=path_color, ls="none")
            ax.plot(*reference[-1], 0, marker="*", ms=6.5, color=path_color, ls="none")
            ax.set(xlim=extent[:2], ylim=extent[2:], zlim=(0, upper))
            ax.set_box_aspect(
                (extent[1] - extent[0], extent[3] - extent[2], upper), zoom=1.10
            )
            ax.view_init(elev=35, azim=-115)
            ax.set_xlabel("x (m)", fontsize=9, labelpad=0)
            ax.set_ylabel("y (m)", fontsize=9, labelpad=-4)
            ax.xaxis.set_major_locator(FixedLocator([0, 10, 20]))
            ax.yaxis.set_major_locator(FixedLocator([-5, 0, 5]))
            ax.zaxis.set_major_locator(FixedLocator([0, 2]))
            ax.tick_params(labelsize=8.5, pad=0, length=2)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.fill = False
                axis._axinfo["grid"]["color"] = theme.color("figure_grid")
                axis._axinfo["grid"]["linewidth"] = 0.45
            title = f"({chr(97+i)}) {spec['label']} · {spec['metrics']['n_obstacles']} obstacles"
            ax.set_title(title, fontsize=9.5, pad=3)
        colorbar_ax = fig.add_axes([0.932, 0.33, 0.012, 0.40])
        colorbar = fig.colorbar(artist, cax=colorbar_ax, ticks=[0, 1, 2, upper])
        colorbar.set_label("Measured z (m)", fontsize=8.5, labelpad=3)
        colorbar.ax.tick_params(labelsize=8, width=0.5, length=2, pad=2)
        colorbar.outline.set_linewidth(0.5)
        handles = [
            Line2D([], [], color=path_color, lw=1.25, label="Executed path"),
            Line2D(
                [],
                [],
                color=reference_color,
                lw=1.05,
                ls=(0, (4, 2)),
                label="Known-map reference",
            ),
            Line2D(
                [], [], color=path_color, marker="o", ms=3, ls="none", label="Start"
            ),
            Line2D(
                [], [], color=path_color, marker="*", ms=6.5, ls="none", label="Goal"
            ),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=4,
            frameon=False,
            fontsize=8.5,
            handlelength=2.2,
            columnspacing=1.4,
            borderaxespad=0.4,
        )
        destination = output / "trajectory_3d_report.png"
        fig.savefig(destination, dpi=300, facecolor=theme.color("white"))
        plt.close(fig)
    return {
        "png": str(destination),
        "cases": [spec["id"] for spec, _, _ in cases],
        "seeds": [result["seed"] for _, result, _ in cases],
        "low_thresholds_m": [
            mapping["metadata"]["ground_threshold_m"] for _, _, mapping in cases
        ],
    }
