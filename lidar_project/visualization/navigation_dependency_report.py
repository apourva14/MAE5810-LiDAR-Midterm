"""Report-sized views of saved navigation dependency measurements.

Rendering reads retained summaries, maps and states only. It does not scan the
scene, run a controller, generate a route, or reconstruct a map.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import theme


POSE_CASES = (
    "baseline", "mount_low", "mount_high", "pitch_up", "pitch_down",
    "yaw_full", "fov_front", "fov_side", "los_occluder",
)
POWER_CASES = (
    "baseline", "reflectance_dark", "reflectance_bright", "target_yaw_45",
    "target_yaw_75", "target_short", "target_tall", "target_near",
    "target_far", "rain", "fog",
)
_LABELS = {
    "baseline": "Baseline", "mount_low": "Mount 0.3 m", "mount_high": "Mount 1.6 m",
    "pitch_up": "Pitch up 15°", "pitch_down": "Pitch down 15°",
    "yaw_full": "Full FOV, yaw 60°", "fov_front": "Front FOV 120°",
    "fov_side": "Side FOV 120°", "los_occluder": "LOS panel added",
    "reflectance_dark": r"$\rho_d=0.02$", "reflectance_bright": r"$\rho_d=0.85$",
    "target_yaw_45": "Target yaw 45°", "target_yaw_75": "Target yaw 75°",
    "target_short": "Target height 0.3 m", "target_tall": "Target height 2.0 m",
    "target_near": "Target x = 9 m", "target_far": "Target x = 13 m",
    "rain": "Rain 50 mm/h", "fog": r"Fog $\kappa=0.6$/m",
    "sampling_sparse": "90 azimuths", "noise_high": "10× receiver noise",
}


def _style(ax):
    ax.tick_params(labelsize=8, length=2.5, width=.6)
    ax.grid(axis="x", color=theme.color("figure_grid"), lw=.5)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(.6)


def _visibility(cases, destination):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(6.5, 2.85))
    grids = fig.add_gridspec(1, 2, width_ratios=(1.45, 1), left=.23,
                            right=.975, bottom=.20, top=.84, wspace=.28)
    axes = [fig.add_subplot(grids[0, 0]), fig.add_subplot(grids[0, 1])]
    blue, orange = theme.PALETTE[:2]
    stage_fields = ("full_fov_opportunities", "target_opportunities", "visible_target_echoes")
    for row, case_id in enumerate(POSE_CASES):
        spec = cases[case_id]
        r, a = spec["replay"], spec["aggregate"]
        counts = np.array([r[field] for field in stage_fields], float) / r["pose_count"]
        axes[0].plot(counts, [row] * 3, color="#BAC2CA", lw=1.1, zorder=1)
        axes[0].plot(counts[0], row-.10, marker="s", ms=4.3, mfc="white", mec="#777777", mew=.8)
        axes[0].plot(counts[1], row, marker="o", ms=3.8, color=blue)
        axes[0].plot(counts[2], row+.10, marker="^", ms=4.0, color=orange)
        gap = a["worst_surface_gap_m"]
        axes[1].plot(gap, row, "o", ms=4.0, color=blue if gap > 0 else orange)
        axes[1].text(.475, row, f"{a['goals']}/{a['runs']}", ha="right", va="center", fontsize=8)
    axes[0].set_yticks(np.arange(len(POSE_CASES)), [_LABELS[c] for c in POSE_CASES], fontsize=8.3)
    axes[1].set_yticks(np.arange(len(POSE_CASES)), [""] * len(POSE_CASES))
    for ax in axes:
        ax.set_ylim(len(POSE_CASES)-.5, -.5)
        _style(ax)
    axes[0].set(xlim=(-5, 195), xticks=(0, 50, 100, 150), xlabel="Target intersections per replay pose")
    axes[1].set(xlim=(-.1, .49), xticks=(0, .2, .4), xlabel="Worst swept surface gap (m)")
    axes[1].axvline(0, color=orange, ls="--", lw=.8)
    axes[1].text(.475, -.84, "Goals", ha="right", fontsize=8)
    axes[0].set_title("(a) Fixed-pose geometric stages", loc="left", fontsize=9, pad=9)
    axes[1].set_title("(b) Closed-loop safety", loc="left", fontsize=9, pad=9)
    for ax in axes:
        ax.xaxis.label.set_fontsize(8.4)
    handles = [
        Line2D([], [], marker="s", ms=4.3, mfc="white", mec="#777777", ls="", label="Full-circle target rays"),
        Line2D([], [], marker="o", ms=3.8, color=blue, ls="", label="Emitted target rays"),
        Line2D([], [], marker="^", ms=4., color=orange, ls="", label="Nearest visible target"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.0),
               frameon=False, ncol=3, fontsize=8, handletextpad=.45, columnspacing=1.1)
    fig.savefig(destination, dpi=300, facecolor="white")
    plt.close(fig)
    return {
        "png": str(destination), "cases": list(POSE_CASES),
        "replay_units": "target intersections per saved baseline pose; pooled over seeds 1–3",
        "navigation_units": "minimum swept gap in m across three complete or terminated runs",
        "stages": list(stage_fields),
    }


def _physical(cases, destination):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(6.5, 3.0))
    # Independent y axes keep each statistic's conditioning explicit.
    grids = fig.add_gridspec(1, 3, width_ratios=(1.12, .82, 1.), left=.215,
                            right=.985, bottom=.19, top=.79, wspace=.65)
    axes = [fig.add_subplot(grids[0, i]) for i in range(3)]
    blue, orange = theme.PALETTE[:2]
    p0 = cases["baseline"]["replay"]["mean_eligible_power_w"]
    for row, case_id in enumerate(POWER_CASES):
        value = cases[case_id]["replay"]["mean_eligible_power_w"] / p0
        axes[0].plot(value, row, "o", ms=4, color=blue)
    axes[0].set_yticks(np.arange(len(POWER_CASES)), [_LABELS[c] for c in POWER_CASES], fontsize=8)
    axes[0].set_ylim(len(POWER_CASES)-.5, -.5)
    axes[0].set_xscale("log")
    axes[0].set(xlim=(.05, 3.5), xticks=(.1, 1, 3), xticklabels=("0.1", "1", "3"))
    axes[0].axvline(1, color="#777777", lw=.7, ls="--")
    axes[0].set_xlabel("Mean eligible target power\n/ baseline mean", fontsize=8.2)
    axes[0].set_title("(a) Target / environment", loc="left", fontsize=9, pad=8)
    count_ids = ("baseline", "sampling_sparse", "fog")
    for row, case_id in enumerate(count_ids):
        r = cases[case_id]["replay"]
        eligible = r["eligible_echoes"] / r["pose_count"]
        detected = r["detected_returns"] / r["pose_count"]
        axes[1].plot([eligible, detected], [row, row], lw=1, color=blue)
        axes[1].plot(eligible, row-.04, marker="o", mfc="white", mec=blue, ms=4.5, mew=.8)
        axes[1].plot(detected, row+.04, "o", color=blue, ms=3.7)
        axes[1].text(.5, row+.37, f"$P_D$={100*r['observed_detection_probability']:.2f}%",
                     transform=axes[1].get_yaxis_transform(), ha="center", va="center", fontsize=8)
    axes[1].set_yticks(range(3), ["Base", "Sparse", "Fog"], fontsize=8)
    axes[1].set(xlim=(0, 200), xticks=(0, 100, 200), ylim=(2.65, -.55))
    axes[1].set_title("(b) Target sampling", loc="left", fontsize=9, pad=8)
    axes[1].set_xlabel("Eligible / detected\nreturns per replay pose", fontsize=8.2)
    noise_ids = ("baseline", "noise_high", "fog")
    for row, case_id in enumerate(noise_ids):
        r = cases[case_id]["replay"]
        axes[2].plot(1000*r["range_rmse_m"], row, "o", color=orange, ms=4.)
        axes[2].plot(1000*r["predicted_range_rmse_m"], row, "|", color="#222222", ms=9, mew=1)
    axes[2].set_yticks(range(3), ["Base", "Noise ×10", "Fog"], fontsize=8)
    axes[2].set(xlim=(9.85, 11.4), xticks=(10, 10.5, 11), ylim=(2.65, -.55))
    axes[2].set_title("(c) Range RMS", loc="left", fontsize=9, pad=8)
    axes[2].set_xlabel("Target RMS (mm)", fontsize=8.2)
    for ax in axes:
        _style(ax)
    handles = [
        Line2D([], [], marker="o", ms=4.5, mfc="white", mec=blue, ls="", label="Eligible echoes"),
        Line2D([], [], marker="o", ms=3.7, color=blue, ls="", label="Detections"),
        Line2D([], [], marker="o", ms=4., color=orange, ls="", label="Measured RMS"),
        Line2D([], [], marker="|", ms=9, color="#222222", ls="", label="Predicted RMS"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=2, frameon=False, fontsize=8.1, columnspacing=1.2, handletextpad=.4)
    fig.savefig(destination, dpi=300, facecolor="white")
    plt.close(fig)
    return {
        "png": str(destination), "power_cases": list(POWER_CASES),
        "sampling_cases": list(count_ids), "range_error_cases": list(noise_ids),
        "power_units": "ratio of pooled mean noiseless eligible target power to baseline",
        "sampling_units": "target returns per saved baseline pose; P_D conditional on eligible target echoes",
        "range_error_units": "mm; measured and predicted RMS conditional on detected target returns",
        "interpretation": "Target yaw/translation/height also change geometric exposure; the pooled power ratios do not isolate incidence or range alone.",
    }


def _maps(directory, cases, destination):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator

    from ..mapping import load_map
    from ..trajectory.storage import load_result

    ids = ("baseline", "fog")
    loaded = [(load_result(directory / cases[c]["representative_run"]),
               load_map(directory / cases[c]["representative_run"])) for c in ids]
    extents = [np.asarray(mapping["metadata"]["extent_m"]) for _, mapping in loaded]
    if any(not np.allclose(extents[0], e) for e in extents[1:]):
        raise ValueError("Navigation report map panels require a common physical ROI")
    references = [np.asarray(result["reference"]["xy"]) for result, _ in loaded]
    if not np.array_equal(references[0], references[1]):
        raise ValueError("Navigation dependency panels require the same saved reference")
    clouds = []
    for result, mapping in loaded:
        xyz = np.asarray(mapping["map_xyz"])
        clouds.append(xyz[xyz[:, 2] > mapping["metadata"]["ground_threshold_m"]])
    upper = np.ceil(max(float(x[:, 2].max()) for x in clouds if len(x)) * 2) / 2
    normalization = Normalize(0, upper)
    path_color, ref_color = theme.color("figure_text"), theme.PALETTE[1]
    extent = extents[0]
    fig = plt.figure(figsize=(6.5, 2.65))
    fig.subplots_adjust(left=.035, right=.915, bottom=.17, top=.91, wspace=.03)
    artist = None
    titles = ("(a) Clear baseline", r"(b) Fog: $\kappa=0.6$ m$^{-1}$")
    for i, ((result, mapping), xyz) in enumerate(zip(loaded, clouds)):
        ax = fig.add_subplot(1, 2, i+1, projection="3d")
        artist = ax.scatter(*xyz.T, c=xyz[:, 2], cmap=theme.colormap("range"),
                            norm=normalization, s=.65, linewidths=0,
                            depthshade=False, rasterized=True)
        actual, reference = np.asarray(result["states"]["xy"]), references[i]
        ax.plot(*reference.T, np.zeros(len(reference)), color=ref_color, lw=1.05, ls=(0, (4, 2)))
        ax.plot(*actual.T, np.zeros(len(actual)), color=path_color, lw=1.25)
        ax.plot(*actual[0], 0, marker="o", ms=3, color=path_color, ls="none")
        ax.plot(*reference[-1], 0, marker="*", ms=6.5, color=path_color, ls="none")
        ax.set(xlim=extent[:2], ylim=extent[2:], zlim=(0, upper))
        ax.set_box_aspect((extent[1]-extent[0], extent[3]-extent[2], upper), zoom=1.1)
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
            axis._axinfo["grid"]["linewidth"] = .45
        ax.set_title(titles[i], fontsize=9.5, pad=3)
        fraction = mapping["metadata"]["observed_cell_fraction"]
        ax.text2D(.5, .98, f"ROI endpoint cells: {100*fraction:.1f}%", transform=ax.transAxes,
                  ha="center", va="top", fontsize=8.5)
    colorbar_ax = fig.add_axes([.932, .33, .012, .4])
    colorbar = fig.colorbar(artist, cax=colorbar_ax, ticks=[0, 1, 2, upper])
    colorbar.set_label("Measured z (m)", fontsize=8.5, labelpad=3)
    colorbar.ax.tick_params(labelsize=8, width=.5, length=2, pad=2)
    colorbar.outline.set_linewidth(.5)
    handles = [
        Line2D([], [], color=path_color, lw=1.25, label="Executed path"),
        Line2D([], [], color=ref_color, lw=1.05, ls=(0, (4, 2)), label="Shared reference"),
        Line2D([], [], color=path_color, marker="o", ms=3, ls="none", label="Start"),
        Line2D([], [], color=path_color, marker="*", ms=6.5, ls="none", label="Goal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               handlelength=2.2, columnspacing=1.4, borderaxespad=.4)
    fig.savefig(destination, dpi=300, facecolor="white")
    plt.close(fig)
    return {
        "png": str(destination), "cases": list(ids),
        "seeds": [result["seed"] for result, _ in loaded],
        "low_thresholds_m": [mapping["metadata"]["ground_threshold_m"] for _, mapping in loaded],
        "seed_observed_fractions": [mapping["metadata"]["observed_cell_fraction"] for _, mapping in loaded],
        "common_extent_m": extent.tolist(), "shared_reference": True,
        "semantics": "Final measured voxel representatives only; no truth surfaces. Low z endpoints hidden only for display. Exact recorded poses.",
    }


def render_navigation_dependency_report(data_dir, output_dir):
    """Create three compact 300 dpi figures from a saved paired study.

    All replay statistics pool the three seed exposures. The map view uses each
    representative run (seed 1 in the supplied study); no truth is plotted.
    """
    import matplotlib.pyplot as plt

    directory, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in summary["cases"]}
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({**theme.MATPLOTLIB, "font.size": 8.5}):
        visibility = _visibility(cases, output / "report_visibility_navigation.png")
        physical = _physical(cases, output / "report_physical_dependencies.png")
        maps = _maps(directory, cases, output / "report_navigation_maps.png")
    return {"seeds": summary["seeds"], "visibility": visibility,
            "physical": physical, "maps": maps}
