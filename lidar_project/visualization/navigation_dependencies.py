"""Render saved, paired navigation-dependency studies without running a sensor.

The two experiment modes are deliberately kept separate: a fixed-pose replay
isolates measurement changes, while a closed-loop run measures their consequences.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import numpy as np

from . import theme
from .mapping import _columns, _jsonable, _take, _truth_meshes


def _number(value, default=np.nan):
    try:
        return float(value) if value is not None and value != "" else float(default)
    except (ValueError, TypeError):
        return float(default)


def _read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: _number(value) if key not in {"status", "case_id", "factor", "label"} else value
                 for key, value in row.items()} for row in csv.DictReader(stream)]


def _get(data, *names, default=np.nan):
    for name in names:
        if name in data and data[name] is not None:
            return _number(data[name], default)
    return float(default)


def _load(directory):
    from ..trajectory.storage import load_result
    from ..mapping import load_map

    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    cases = []
    for spec in summary["cases"]:
        run_dir = directory / spec["representative_run"]
        replay_dir = directory / spec["representative_replay"]
        result, mapping = load_result(run_dir), load_map(run_dir)
        if not result.get("scans_available", False):
            raise ValueError(f"{spec['id']} has no retained 3D scans")
        replay_json = replay_dir / "replay.json"
        replay_meta = json.loads(replay_json.read_text()) if replay_json.exists() else {}
        cases.append(dict(spec=spec, result=result, mapping=mapping,
                          traces=_read_rows(replay_dir / "traces.csv"), replay_metadata=replay_meta,
                          replay_dir=replay_dir))
    if not cases:
        raise ValueError("The navigation dependency study has no cases")
    return summary, cases


def _base(cases, spec):
    name = spec.get("comparison_baseline", "baseline")
    return next((item for item in cases if item["spec"]["id"] == name), cases[0])


def _factor_colors(cases):
    factors = list(dict.fromkeys(item["spec"]["factor"] for item in cases))
    return {factor: theme.PALETTE[i % len(theme.PALETTE)] for i, factor in enumerate(factors)}


def _overview(cases, output):
    """All outcomes, including failed prefixes, in a common set of physical units."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = _factor_colors(cases)
    n = len(cases)
    fig, axes = plt.subplots(1, 5, figsize=(15.2, max(5.0, .34 * n + 1.8)),
                             sharey=True, layout="constrained")
    y = np.arange(n)
    labels = [item["spec"]["label"] for item in cases]
    values = [[], [], [], [], []]
    for item in cases:
        a = item["spec"]["aggregate"]
        values[0].append(100 * _get(a, "goals", default=0) / max(1, _get(a, "runs", default=1)))
        values[1].append(_get(a, "worst_surface_gap_m"))
        values[2].append(_get(a, "mean_elapsed_s"))
        values[3].append(_get(a, "mean_path_length_m"))
        values[4].append(100 * _get(a, "mean_observed_fraction"))
    titles = ["Goal completion", "Worst swept gap", "Mean end time", "Mean path length", "Observed endpoint cells"]
    xlabels = ["Runs reaching goal (%)", "Surface gap (m)", "Termination time (s)", "Path length (m)", "Common region observed (%)"]
    for col, (ax, value, title, xlabel) in enumerate(zip(axes, values, titles, xlabels)):
        for i, item in enumerate(cases):
            ax.plot(value[i], i, "o", ms=5.4, color=colors[item["spec"]["factor"]])
            if col == 0:
                a = item["spec"]["aggregate"]
                ax.annotate(f"{int(a['goals'])}/{int(a['runs'])}", (value[i], i), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=8)
        ax.set_title(title, loc="left", fontsize=10, pad=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_yticks(y, labels, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="both", color=theme.color("figure_grid"), lw=.5)
        ax.set_axisbelow(True)
    axes[0].set_xlim(-7, 128)
    axes[0].invert_yaxis()
    axes[1].axvline(0, color=theme.color("collision"), lw=1, ls="--")
    axes[4].set_xlim(0, 105)
    handles = [Line2D([], [], marker="o", ls="", color=color, label=factor.replace("_", " ").capitalize())
               for factor, color in colors.items()]
    fig.legend(handles=handles, loc="outside upper center", ncol=min(5, len(handles)),
               fontsize=8, frameon=False)
    fig.supxlabel("Closed-loop navigation · every saved seed included · failures contribute only their actual recorded prefix\n"
                  "Endpoint-observed area is sampling extent, not map accuracy or free-space coverage.", fontsize=9)
    path = output / "navigation_dependency_outcomes.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)
    return str(path.resolve())


def _replay_fields(spec):
    r = spec.get("replay", {})
    poses = _get(r, "pose_count", "snapshots", "total_poses", "total_samples", default=0)
    def per_pose(name):
        return _get(r, name) / poses if poses > 0 else np.nan
    return dict(
        eligible=per_pose("eligible_echoes"),
        detected=per_pose("detected_returns"),
        pd=_get(r, "observed_detection_probability", "observed_pd", "detection_probability"),
        expected_pd=_get(r, "expected_detection_probability", "expected_pd"),
        power=_get(r, "mean_eligible_power_w"),
        range_rmse=_get(r, "range_rmse_m", "detected_range_rmse_m", "conditional_range_rmse_m"),
        predicted_rmse=_get(r, "predicted_range_rmse_m", "expected_range_rmse_m"),
        target_hits=_get(r, "mean_target_detected_returns_per_scan", "mean_target_returns", "target_returns_per_pose"),
        target_eligible=_get(r, "mean_target_eligible_echoes_per_scan", "mean_target_eligible_echoes"),
    )


def _replay_overview(cases, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = _factor_colors(cases)
    height = max(5.1, .34 * len(cases) + 1.5)
    fig, axes = plt.subplots(1, 4, figsize=(13.4, height), sharey=True, layout="constrained")
    y = np.arange(len(cases))
    labels = [item["spec"]["label"] for item in cases]
    for i, item in enumerate(cases):
        r, color = _replay_fields(item["spec"]), colors[item["spec"]["factor"]]
        axes[0].plot([r["eligible"], r["detected"]], [i, i], color=color, lw=1)
        axes[0].plot(r["eligible"], i, "o", mfc="white", mec=color, ms=5)
        axes[0].plot(r["detected"], i, "o", color=color, ms=4)
        axes[1].plot(100 * r["expected_pd"], i, "|", color="black", ms=10, mew=1)
        axes[1].plot(100 * r["pd"], i, "o", color=color, ms=4)
        base = _replay_fields(_base(cases, item["spec"])["spec"])
        ratio = r["power"] / base["power"] if base["power"] > 0 else np.nan
        axes[2].plot(ratio, i, "o", color=color, ms=4)
        axes[3].plot(1000 * r["range_rmse"], i, "o", color=color, ms=4)
        if np.isfinite(r["predicted_rmse"]):
            axes[3].plot(1000 * r["predicted_rmse"], i, "|", color="black", ms=10)
    titles = ["Target echoes and detections", "Detection given eligible target", "Mean eligible target power", "Detected target range error"]
    xlabels = ["Mean per baseline pose", "Observed / expected (%)", "Ratio to stated baseline", "Conditional RMS (mm)"]
    for ax, title, label in zip(axes, titles, xlabels):
        ax.set_title(title, loc="left", fontsize=10, pad=9)
        ax.set_xlabel(label, fontsize=9)
        ax.set_yticks(y, labels, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(color=theme.color("figure_grid"), lw=.5)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    axes[1].set_xlim(-4, 104)
    axes[2].axvline(1, color=theme.color("ground"), lw=.8, ls="--")
    fig.legend(handles=[Line2D([], [], marker="o", mfc="white", color="black", ls="", label="Eligible echo"),
                        Line2D([], [], marker="o", color="black", ls="", label="Measured detection / RMS"),
                        Line2D([], [], marker="|", color="black", ls="", ms=10, label="Analytical expectation")],
               loc="outside upper center", ncol=3, fontsize=8, frameon=False)
    fig.supxlabel("Fixed-pose replay · target-conditioned statistics at identical baseline robot poses\n"
                  "Eligibility precedes noisy thresholding. Detected-only range RMS uses its recorded detection denominator.", fontsize=9)
    path = output / "navigation_dependency_measurements.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)
    return str(path.resolve())


def _case_payload(item):
    spec, result, mapping = item["spec"], item["result"], item["mapping"]
    meta = mapping["metadata"]
    points = np.asarray(mapping["map_xyz"])
    selection = _take(len(points), 18000)
    cloud = _columns(points[selection], np.asarray(mapping["map_intensity"])[selection])
    cloud["step"] = np.asarray(mapping["map_first_seen"])[selection]
    extent = meta.get("roi_extent_m", meta["extent_m"])
    scans = []
    for scan in result["scans"]:
        xyz, intensity = np.asarray(scan["xyz_world"]), np.asarray(scan["intensity"])
        inside = ((xyz[:, 0] >= extent[0]) & (xyz[:, 0] < extent[1]) &
                  (xyz[:, 1] >= extent[2]) & (xyz[:, 1] < extent[3]))
        candidates = np.flatnonzero(inside)
        keep = candidates[_take(len(candidates), 550)]
        scans.append(_columns(xyz[keep], intensity[keep]))
    return dict(id=spec["id"], label=spec["label"], factor=spec["factor"],
                baseline=spec.get("comparison_baseline", "baseline"), description=spec.get("description", ""),
                aggregate=spec["aggregate"], replay=spec.get("replay", {}), replay_fields=_replay_fields(spec),
                traces=item["traces"], states=result["states"], scans=scans, cloud=cloud,
                scan_counts=[len(scan["xyz_world"]) for scan in result["scans"]],
                map_counts=np.cumsum(np.bincount(np.asarray(mapping["map_first_seen"], int), minlength=len(scans))),
                extent=extent, ground_threshold=meta.get("ground_threshold_m", .12),
                reference=result["reference"]["xy"], meshes=_truth_meshes(result),
                radius=result["config"]["robot"]["radius_m"],
                body_height=min(.3, result["config"]["robot"]["height_m"]),
                sensor_height=result["config"]["mount_height_m"], mount=result["config"].get("sensor_mount", {}),
                sensor_enabled=result["config"].get("lidar_enabled", True), status=result["status"], seed=result["seed"])


def _visibility(cases, output):
    """Show the successive geometry/visibility/range gates with honest denominators."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(10.6, max(5.0, .34 * len(cases) + 1.6)),
                             layout="constrained")
    colors = _factor_colors(cases)
    for i, item in enumerate(cases):
        r = item["spec"]["replay"]
        ratios = [("target_opportunities", "full_fov_opportunities"),
                  ("visible_target_echoes", "target_opportunities"),
                  ("eligible_echoes", "visible_target_echoes")]
        for ax, (num, den) in zip(axes, ratios):
            d = _get(r, den)
            ratio = 100 * _get(r, num) / d if d > 0 else np.nan
            ax.plot(ratio, i, "o", ms=4.5, color=colors[item["spec"]["factor"]])
    labels = [item["spec"]["label"] for item in cases]
    titles = ["FOV retains target intersections", "LOS retains nearest target echoes", "Range gate retains visible echoes"]
    xlabels = ["Actual / full-FOV opportunities (%)", "Visible / emitted target intersections (%)", "Eligible / nearest visible target echoes (%)"]
    for ax, title, xlabel in zip(axes, titles, xlabels):
        ax.set_title(title, loc="left", fontsize=9.5, pad=9)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_yticks(np.arange(len(cases)), labels, fontsize=9)
        ax.set_xlim(-5, 105)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(color=theme.color("figure_grid"), lw=.5)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.supxlabel("Fixed-pose replay · same emitted-ray sampling grid · opaque occluders precede range and detection gates\n"
                  "The denominators differ by stage. An empty denominator is undefined, so no point is drawn.", fontsize=9)
    path = output / "navigation_dependency_visibility.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)
    return str(path.resolve())


def _viewer(summary, cases, output):
    from plotly.offline import get_plotlyjs
    from plotly.colors import get_colorscale

    payload = dict(cases=[_case_payload(item) for item in cases], settings=summary.get("settings", {}),
                   definitions=summary.get("definitions", {}), limitations=summary.get("limitations", []),
                   seeds=summary.get("seeds", []), height_scale=get_colorscale(theme.colormap("range")),
                   colors={"background": theme.color("white"), "text": theme.color("figure_text"),
                           "grid": theme.color("figure_grid"), "baseline": theme.PALETTE[0],
                           "variant": theme.PALETTE[1], "scan": theme.PALETTE[3],
                           "truth": theme.color("ground"), "path": theme.color("figure_text")},
                   font=theme.THEME["web"]["font_family"])
    data = json.dumps(_jsonable(payload), allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    replacements = {"__STYLE__": theme.stylesheet("navigation_dependencies.css"), "__PLOTLY__": get_plotlyjs(),
                    "__DATA__": data, "__LOGIC__": theme.asset("static/navigation_dependencies.js")}
    page = re.sub(r"__[A-Z_]+__", lambda match: replacements[match.group(0)],
                  theme.asset("templates/navigation_dependencies.html"))
    path = output / "navigation_dependencies.html"
    path.write_text(page, encoding="utf-8")
    return str(path.resolve())


def render_navigation_dependencies(data_dir, output_dir, *, interactive=True):
    """Read saved studies and produce static evidence plus an offline 3D viewer."""
    import matplotlib.pyplot as plt

    directory, output = Path(data_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    summary, cases = _load(directory)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(theme.MATPLOTLIB):
        outputs = dict(outcomes=_overview(cases, output), measurements=_replay_overview(cases, output),
                       visibility=_visibility(cases, output))
        if interactive:
            outputs["viewer"] = _viewer(summary, cases, output)
    return outputs
