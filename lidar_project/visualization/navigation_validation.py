"""Publication figures and an offline presentation of saved validation data.

This module reads prepared geometry, Monte Carlo summaries and navigation
outcomes. It performs no ray casting, scanning, planning or map construction.
"""
from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from . import theme


_LAYOUTS = ("open", "cluttered", "narrow")
_LAYOUT_LABELS = {"open": "Open", "cluttered": "Cluttered", "narrow": "Narrow passage"}
_LAYOUT_COLORS = ("#0072B2", "#D55E00", "#009E73")
_ABLATIONS = ("baseline", "mount_high", "sampling_sparse", "fov_front", "fov_side", "lidar_disabled")
_CASE_LABELS = {
    "baseline": "Baseline", "mount_high": "Mount 1.6 m", "sampling_sparse": "90 azimuths",
    "fov_front": "Front FOV 120°", "fov_side": "Side FOV 120°", "lidar_disabled": "LiDAR disabled",
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _axes_style(ax, axis="both"):
    ax.tick_params(labelsize=8, width=.6, length=2.7)
    ax.grid(axis=axis, color="#E6E8EB", lw=.5)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(.6)


def _sector_lines(position, rotation, horizontal, vertical, radius):
    """Display-only geometry of the stated angular FOV; no visibility queries."""
    def world(azimuth, elevation):
        a, e = np.broadcast_arrays(np.radians(azimuth), np.radians(elevation))
        local = np.stack((np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)), axis=-1)
        return position + radius * local @ rotation.T
    azimuth = np.linspace(*horizontal, 101)
    elevation = np.linspace(*vertical, 41)
    lines = [world(azimuth, vertical[0]), world(azimuth, vertical[1]),
             world(horizontal[0], elevation), world(horizontal[1], elevation)]
    for a in horizontal:
        for e in vertical:
            lines.append(np.stack((position, world(a, e))))
    return lines, world


def _fov_los(data, destination):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    sensor = data["sensor"]
    origin = np.asarray(sensor["position"], float)
    rotation = np.asarray(sensor["rotation_world_from_sensor"], float)
    horizontal, vertical, radius = data["hfov_deg"], data["vfov_deg"], data["display_range_m"]
    blue, red = "#0072B2", "#B44440"
    target_color, blocker_color = "#E9B965", "#89929A"
    fig = plt.figure(figsize=(6.5, 2.8))
    fig.subplots_adjust(left=.005, right=.995, bottom=.23, top=.88, wspace=.035)
    lines, world = _sector_lines(origin, rotation, horizontal, vertical, radius)
    # Shared physical bounds include every displayed FOV corner in both panels.
    corners = np.vstack([line for line in lines])
    limits = np.stack((corners.min(axis=0), corners.max(axis=0)), axis=1)
    limits[0] = (origin[0]-.4, limits[0,1]+.15)
    limits[1] += [-.15,.15]
    limits[2] = (min(-.9, limits[2,0]), 2.25)
    for panel_index, panel in enumerate(data["panels"]):
        ax = fig.add_subplot(1,2,panel_index+1,projection="3d")
        for line in lines:
            ax.plot(*line.T, color="#5B8EAF", lw=.65, alpha=.7)
        edge = world(np.linspace(*horizontal, 41), vertical[1])
        polygons = [[origin, edge[i], edge[i+1]] for i in range(len(edge)-1)]
        ax.add_collection3d(Poly3DCollection(polygons, facecolor="#9CC7DF", alpha=.055, edgecolor="none"))
        ground = [[(limits[0,0],limits[1,0],0),(limits[0,1],limits[1,0],0),
                   (limits[0,1],limits[1,1],0),(limits[0,0],limits[1,1],0)]]
        ax.add_collection3d(Poly3DCollection(ground,facecolor="#D7DADD",alpha=.09,edgecolor="none"))
        for item in panel["objects"]:
            vertices = np.asarray(item["vertices"], float)
            faces = [vertices[index] for index in item["faces"]]
            role = item["role"]
            face = target_color if role == "target" else blocker_color
            ax.add_collection3d(Poly3DCollection(faces, facecolor=face, edgecolor="#424B53",
                                               linewidth=.45, alpha=.54 if role == "occluder" else .46))
            label_at = np.asarray(item.get("label_at", vertices.mean(axis=0)), float)
            if role == "occluder":
                ax.text2D(.39,.76,"Opaque panel",transform=ax.transAxes,fontsize=7.8,
                          ha="center",va="bottom",color="#222222")
            else:
                ax.text(*label_at,"Target",fontsize=7.8,ha="center",va="bottom",color="#222222")
        for record in panel["rays"]:
            target = np.asarray(record["target_point"], float)
            nearest = np.asarray(record["nearest_point"], float)
            if record["visible"]:
                ax.plot(*np.stack((origin, target)).T, color=blue, lw=.85, alpha=.85)
            else:
                ax.plot(*np.stack((origin, nearest)).T, color=red, lw=.85, alpha=.85)
                ax.plot(*np.stack((nearest, target)).T, color=red, lw=1.0, ls=(0, (3, 2)))
                ax.scatter(*nearest, s=7, marker="x", color=red, linewidth=.7, depthshade=False)
        ax.scatter(*origin, s=20, color="#111111", marker="o", depthshade=False)
        ax.text(*(origin+[-.28,0,-.34]), "LiDAR", fontsize=7.8, ha="center", va="top")
        for axis_index, letter in enumerate(("xₛ", "yₛ", "zₛ")):
            direction = rotation[:, axis_index] * .55
            ax.quiver(*origin, *direction, color="#333333", lw=.65, arrow_length_ratio=.15)
            label_offset = [-.22,.12,.06] if axis_index == 2 else [0,0,.13]
            ax.text(*(origin+direction*1.6+label_offset), letter, fontsize=7)
        ax.set(xlim=limits[0],ylim=limits[1],zlim=limits[2])
        ax.set_box_aspect(np.diff(limits,axis=1).ravel(),zoom=1.14)
        ax.view_init(elev=26,azim=-110)
        ax.set_xlabel("x (m)",fontsize=8,labelpad=-5)
        ax.set_ylabel("y (m)",fontsize=8,labelpad=-4)
        ax.set_zlabel("z (m)",fontsize=8,labelpad=-7)
        ax.set_xticks((10,12,14))
        ax.set_yticks((0,2,4,6))
        ax.set_zticks((0,2))
        ax.tick_params(labelsize=7.4,pad=0)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.pane.fill=False
            axis._axinfo["grid"]["color"]="#E6E8EB"
            axis._axinfo["grid"]["linewidth"]=.4
        audit=panel["audit"]
        ax.set_title(f"({'ab'[panel_index]}) {panel['label']}",fontsize=9,pad=5)
        ax.text2D(.5,.99,f"{audit['visible_target_echoes']}/{audit['target_opportunities']} visible target rays; "
                  f"{audit['blocked_target_echoes']} blocked",transform=ax.transAxes,fontsize=8,
                  ha="center",va="top")
    handles=[Patch(facecolor="#D9EAF4",edgecolor="#5B8EAF",label="Nominal FOV"),
             Line2D([],[],color=blue,lw=1.1,label="Visible target ray"),
             Line2D([],[],color=red,lw=1.1,label="Nearest opaque hit"),
             Line2D([],[],color=red,lw=1.1,ls="--",label="Blocked continuation")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,-.005),ncol=4,
               fontsize=7.4,frameon=False,columnspacing=1.,handlelength=1.8)
    fig.savefig(destination,dpi=300,facecolor="white")
    plt.close(fig)
    return {"png":str(destination),"source":data["source"],
            "panels":{panel["id"]:panel["audit"] for panel in data["panels"]},
            "display_range_m":radius,"physical_max_range_m":data["max_range_m"],
            "semantics":"Prepared scene truth and geometrically audited rays, not measured points; FOV display range is truncated. All rays counted, representative subset drawn."}


def _threshold(summary, destination):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    settings = summary["settings"]
    pose_ids = list(dict.fromkeys(row["pose_id"] for row in settings))
    threshold_estimates = [row["mean_snr"]-NormalDist().inv_cdf(row["expected_pd"])
                           for row in settings if .05 < row["expected_pd"] < .95]
    gamma = float(summary.get("snr_threshold", np.median(threshold_estimates)))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2), gridspec_kw={"width_ratios": (1.65, 1.)})
    fig.subplots_adjust(left=.08, right=.985, bottom=.24, top=.80, wspace=.33)
    x = np.linspace(min(r["mean_snr"] for r in settings), max(r["mean_snr"] for r in settings), 300)
    probability = np.array([.5*math.erfc((gamma-value)/math.sqrt(2)) for value in x])
    axes[0].plot(x, probability, color="#222222", lw=1.1)
    axes[0].axvspan(gamma-1.645, gamma+1.645, color="#EDF1F4", zorder=-1)
    axes[0].axvline(gamma, color="#68788B", lw=.7, ls="--")
    handles = [Line2D([], [], color="#222222", lw=1.1, label="Gaussian prediction")]
    for index, pose_id in enumerate(pose_ids):
        rows = sorted((r for r in settings if r["pose_id"] == pose_id), key=lambda r: r["mean_snr"])
        xx = np.array([r["mean_snr"] for r in rows])
        yy = np.array([r["observed_pd"] for r in rows])
        expected = np.array([r["expected_pd"] for r in rows])
        error = np.array([[max(0., r["observed_pd"]-r["wilson_low"]) for r in rows],
                          [max(0., r["wilson_high"]-r["observed_pd"]) for r in rows]])
        color = _LAYOUT_COLORS[index % len(_LAYOUT_COLORS)]
        marker = ("o", "s", "^")[index % 3]
        offset = (index-(len(pose_ids)-1)/2)*.075
        for ax, values in zip(axes, (yy, yy-expected)):
            ax.errorbar(xx+offset, values, yerr=error, fmt=marker, ms=3.3, color=color,
                        mew=.6, mfc="white", lw=.7, capsize=1.8, capthick=.6)
        handles.append(Line2D([], [], marker=marker, ms=3.5, color=color, mfc="white", ls="",
                              label=rows[0].get("pose_label", rows[0].get("label", str(pose_id)))))
    axes[0].set(xlim=(x[0]-.3, x[-1]+.3), ylim=(-.025, 1.025),
                yticks=(0, .25, .5, .75, 1), ylabel="Detection probability", xlabel="Mean received SNR")
    axes[1].axhline(0, color="#222222", lw=.8)
    axes[1].set(xlim=(x[0]-.3, x[-1]+.3), ylabel="Observed − predicted", xlabel="Mean received SNR")
    axes[1].yaxis.label.set_fontsize(8)
    for ax in axes:
        ax.set_xticks((6, 8, 10, 12, 14))
        ax.xaxis.label.set_fontsize(8.2)
        ax.yaxis.label.set_fontsize(8.2)
        _axes_style(ax)
    axes[0].set_title("(a) Detection across the threshold", loc="left", fontsize=9, pad=7)
    axes[1].set_title("(b) Calibration residual", loc="left", fontsize=9, pad=7)
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7.8, frameon=False,
               columnspacing=1., handletextpad=.4, bbox_to_anchor=(.5, 1.01))
    fig.savefig(destination, dpi=300, facecolor="white")
    plt.close(fig)
    return {"png": str(destination), "pose_ids": pose_ids, "settings": len(settings),
            "trials": sum(row["trials"] for row in settings),
            "intervals": "Pointwise 95% Wilson binomial intervals; horizontal pose offsets are display only",
            "prediction": "Phi(mean_SNR - SNR_threshold); shaded band is predicted 5–95% transition"}


def _layout_id(case):
    value = case["layout_id"]
    return "narrow" if value in ("narrow_passage", "narrow_corridor") else value


def _robustness(summary, destination):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cases = summary["cases"]
    layouts = [layout for layout in _LAYOUTS if any(_layout_id(c) == layout for c in cases)]
    other = [c["layout_id"] for c in cases if _layout_id(c) not in layouts]
    layouts.extend(dict.fromkeys(other))
    lookup = {(_layout_id(row), row["case_id"]): row for row in cases}
    fog = (("baseline", .0002), ("fog_010", .10), ("fog_030", .30), ("fog_060", .60),
           ("fog_120", 1.20), ("fog_300", 3.), ("fog_1000", 10.))
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.8), gridspec_kw={"width_ratios": (1., 1., 1.22)})
    fig.subplots_adjust(left=.075, right=.987, top=.79, bottom=.21, wspace=.47)
    handles = []
    for index, layout in enumerate(layouts):
        color = _LAYOUT_COLORS[index % len(_LAYOUT_COLORS)]
        marker = ("o", "s", "^")[index % 3]
        label = _LAYOUT_LABELS.get(layout, layout.replace("_", " ").title())
        handles.append(Line2D([], [], marker=marker, color=color, ms=3.7, lw=.9, label=label))
        rows = [lookup[(layout, case_id)] for case_id, _ in fog]
        x = np.array([kappa for _, kappa in fog])
        y = np.array([row["goal_fraction"] for row in rows])
        ci = np.array([row["goal_fraction_wilson95"] for row in rows]).T
        error = np.maximum(0, np.stack((y-ci[0], ci[1]-y)))
        xdisplay = x * (1+(index-(len(layouts)-1)/2)*.035)
        axes[0].errorbar(xdisplay, y, yerr=error, color=color, marker=marker, ms=3.3,
                         mfc="white", lw=.9, capsize=1.5, capthick=.6)
        gaps = [row["all_run_prefix"]["swept_min_surface_gap_m"] for row in rows]
        median = np.array([gap["median"] for gap in gaps])
        lower, upper = np.array([gap["q25"] for gap in gaps]), np.array([gap["q75"] for gap in gaps])
        axes[1].plot(x, median, color=color, marker=marker, ms=3.3, mfc="white", lw=.9)
        axes[1].fill_between(x, lower, upper, color=color, alpha=.12, linewidth=0)
        for rownum, case_id in enumerate(_ABLATIONS):
            row = lookup[(layout, case_id)]
            low, high = row["goal_fraction_wilson95"]
            mean = row["goal_fraction"]
            axes[2].errorbar(mean, rownum+(index-(len(layouts)-1)/2)*.18,
                             xerr=[[max(0, mean-low)], [max(0, high-mean)]], fmt=marker,
                             color=color, mfc="white", ms=3.2, lw=.7, capsize=1.2)
    for ax in axes:
        _axes_style(ax)
    axes[0].set(xlim=(-.02, 11.5), ylim=(-.04, 1.045), yticks=(0, .5, 1),
                ylabel="Goal fraction", xlabel=r"Extinction $\kappa$ (m$^{-1}$)")
    axes[1].axhline(0, color="#B44440", lw=.8, ls="--")
    axes[1].set(xlim=(-.02, 11.5), ylabel="Swept surface gap (m)",
                xlabel=r"Extinction $\kappa$ (m$^{-1}$)")
    for ax in axes[:2]:
        ax.set_xscale("symlog",linthresh=.1,linscale=.65)
        ax.set_xticks((0, .1, .6, 3, 10), ("0", ".1", ".6", "3", "10"))
    axes[2].set(xlim=(-.04, 1.045), xticks=(0, .5, 1),
                yticks=range(len(_ABLATIONS)), yticklabels=["Base", "High mount", "Sparse az.", "Front FOV", "Side FOV", "Off"],
                ylim=(len(_ABLATIONS)-.5, -.5), xlabel="Goal fraction")
    axes[2].tick_params(axis="y", labelsize=7.6, pad=2)
    for ax in axes:
        ax.xaxis.label.set_fontsize(8.1)
        ax.yaxis.label.set_fontsize(8.1)
    for ax, title in zip(axes, ("(a) Optical stress: goals", "(b) Optical stress: gap", "(c) Sensor settings")):
        ax.set_title(title, fontsize=8.7, loc="left", pad=7)
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8.2,
               bbox_to_anchor=(.5, 1.01), columnspacing=1.9)
    fig.text(.5, .025, "20 noise seeds per condition and scene  ·  Goal bars: pointwise 95% Wilson CI  ·  Gap: median and IQR, all runs",
             ha="center", fontsize=7.4)
    fig.savefig(destination, dpi=300, facecolor="white")
    plt.close(fig)
    return {"png": str(destination), "layouts": layouts, "cases": len(cases),
            "runs": sum(row["runs"] for row in cases), "fog_extinction_per_m": [x for _, x in fog],
            "intervals": "Pointwise Wilson 95% binomial intervals; no simultaneous coverage claim",
            "safety": "Median and interquartile range of per-run swept minimum surface gap; collisions retained",
            "x_offsets": "Tiny fog success offsets distinguish layouts; physical settings unchanged"}


def _image_data(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _offline_html(diagram, threshold, robustness, figures, destination):
    rows = []
    for row in robustness["cases"]:
        low, high = row["goal_fraction_wilson95"]
        gap = row["all_run_prefix"]["swept_min_surface_gap_m"]
        completed = row["successful_runs"]
        completed_time = completed["elapsed_s"]["median"]
        completed_path = completed["path_length_m"]["median"]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            _LAYOUT_LABELS.get(_layout_id(row), row["layout_id"]), row["label"],
            f"{row['goals']}/{row['runs']}", f"{100*low:.1f}–{100*high:.1f}%",
            row["collisions"], row["timeouts"], f"{gap['median']:.3f}",
            f"{gap['q25']:.3f}–{gap['q75']:.3f}", f"{gap['min']:.3f}",
            f"{completed_time:.1f}" if completed_time is not None else "—",
            f"{completed_path:.3f}" if completed_path is not None else "—",
        )) + "</tr>")
    images = {name: _image_data(figures[name]["png"]) for name in ("fov_los", "threshold", "robustness")}
    trial_count = sum(row["trials"] for row in threshold["settings"])
    run_count = sum(row["runs"] for row in robustness["cases"])
    calibration = threshold["aggregate"]
    text = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LiDAR navigation validation</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f5f7;color:#202a34;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1080px;margin:auto;padding:40px 28px 64px}}h1{{font-size:32px;line-height:1.2;margin:0 0 14px}}h2{{font-size:22px;line-height:1.3;margin:0 0 12px}}p{{max-width:83ch}}.lead{{color:#4d5e6e;font-size:18px}}.facts{{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}}.facts span{{background:#e5edf4;border:1px solid #cbd9e5;padding:7px 13px;border-radius:5px}}section{{background:white;border:1px solid #dce2e8;border-radius:8px;margin:25px 0;padding:26px}}img{{display:block;width:100%;height:auto;margin:18px 0}}figcaption,.note{{font-size:14px;color:#536170}}table{{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}}td,th{{text-align:left;padding:9px 8px;border-bottom:1px solid #e2e7ec}}th{{background:#edf2f6}}.scroll{{overflow:auto}}details summary{{cursor:pointer;font-weight:600}}a{{color:#006b9e}}@media(max-width:650px){{main{{padding:25px 12px}}section{{padding:18px 10px}}h1{{font-size:27px}}}}
</style></head><body><main>
<h1>LiDAR navigation validation</h1>
<p class="lead">A physical dependency should first change a measurement, then alter navigation only when that information matters to the task. These checks separate those two claims.</p>
<div class="facts"><span>3 navigation layouts</span><span>{run_count:,} closed-loop runs</span><span>{trial_count:,} threshold trials</span><span>Offline figures and data tables</span></div>
<section><h2>1. Coverage and visibility are separate</h2>
<p>The blue boundaries show the configured nominal angular field of view. A target-directed ray can remain inside that coverage and still be blocked: the nearest opaque surface terminates its line of sight. Dashed red continuations identify where the ray would reach the target if the foreground panel were absent.</p>
<figure><img src="{images['fov_los']}" alt="Paired three-dimensional nominal LiDAR coverage at the same navigation pose, without and with a foreground opaque panel, visible rays and blocked target continuations"><figcaption>Both panels use the same recorded navigation pose at t = {diagram['source']['time_s']:.1f} s. The configured angular coverage is −60° to 60° azimuth and −20° to 10° elevation; the actual range gate is {diagram['min_range_m']:g}–{diagram['max_range_m']:g} m. The displayed envelope ends at {diagram['display_range_m']:g} m for readability. Only the relevant target and panel geometry and representative rays are drawn; all counts use every emitted ray and the full scene's nearest-hit geometry. This is explanatory ground-truth geometry, not a noisy measured point cloud.</figcaption></figure>
</section><section><h2>2. Check detection where it is uncertain</h2>
<p>At three recorded navigation poses, a real eligible target ray is replayed with independent receiver noise. Extinction is solved to place its mean received SNR at 11 settings from 6 to 14. With a threshold of 10, Gaussian receiver noise predicts P<sub>D</sub> = Φ(mean SNR − 10). Each setting has 1,000 trials.</p>
<figure><img src="{images['threshold']}" alt="Detection probability near the threshold with Gaussian prediction, Monte Carlo estimates, Wilson intervals and prediction residuals"><figcaption>Markers and pointwise 95% Wilson intervals are simulation estimates; the continuous line is the analytical prediction. Small horizontal offsets distinguish poses. Intervals do not provide simultaneous 95% coverage of all settings.</figcaption></figure>
<p><strong>Numerical check:</strong> probability RMSE = {calibration['calibration_rmse']:.5f}; maximum absolute probability error = {calibration['max_abs_detection_residual']:.5f} ({100*calibration['max_abs_detection_residual']:.2f} percentage points). The model lies inside {calibration['pointwise_wilson_contains_model']} of {calibration['settings']} pointwise intervals. Across {calibration['detections']:,} detected echoes, conditional range RMS is {1000*calibration['conditional_range_rmse_m']:.2f} mm, versus {1000*calibration['predicted_conditional_range_rmse_m']:.2f} mm predicted by the model.</p>
<p class="note">This is conditional detection of an already visible, range-eligible target echo. It does not test whether geometry, occlusion, or angular sampling produces such an echo in the first place.</p>
</section><section><h2>3. Test whether measurement losses matter to navigation</h2>
<p>Each of three layouts uses its own fixed incomplete-map reference route. The controller, robot and task settings stay fixed while optical extinction or one sensor setting changes. Twenty paired receiver-noise seeds are run for every condition. Failures remain in all-run safety statistics. The largest extinction coefficients are deliberately severe optical stress tests and are not calibrated fog or rain conditions.</p>
<figure><img src="{images['robustness']}" alt="Goal completion and swept clearance over optical extinction and sensor ablations in three navigation layouts"><figcaption>Goal bars are pointwise 95% Wilson intervals. Safety curves show the median and interquartile range of each run's minimum swept surface gap; zero or negative gap means contact or overlap. High mount is 1.6 m; sparse sampling uses 90 azimuths; front and side coverage are 120°; off disables LiDAR. Run prefixes ending in collision or timeout are not completed routes.</figcaption></figure>
<details><summary>All condition outcomes and uncertainty</summary><p class="note">Gap summaries include all completed and terminated run prefixes. Time and path medians use successful runs only; an em dash means no run reached the goal. Conditioning on success excludes failures and must be interpreted together with the goal fraction. Scroll the table horizontally to see completed-route time and path.</p><div class="scroll"><table><thead><tr><th>Scene</th><th>Condition</th><th>Goals</th><th>Wilson 95% CI</th><th>Collisions</th><th>Timeouts</th><th>Median gap (m)</th><th>Gap IQR (m)</th><th>Worst gap (m)</th><th>Completed median time (s)</th><th>Completed median path (m)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></details>
</section><section><h2>What these tests establish</h2>
<p>The geometry diagram explains the FOV/LOS mechanism; near-threshold trials test calibration against the assumed stochastic model; the multi-layout navigation study tests task consequences under a frozen controller. Agreement establishes internal consistency and simulated task robustness under these conditions.</p>
<p class="note">The sensor coefficients and weather model remain illustrative. Exact poses, static obstacles, snapshot scans and three hand-designed layouts limit external generalization. Noise seeds repeat stochastic measurements; they are not independent new environments. A 20/20 result is not proof of perfect reliability.</p>
</section></main></body></html>'''
    destination.write_text(text, encoding="utf-8")
    return str(destination)


def render_navigation_validation(diagram_path, threshold_dir, robustness_dir, output_dir):
    """Render three 300 dpi figures and one self-contained offline HTML file."""
    import matplotlib.pyplot as plt

    diagram = _read(diagram_path)
    threshold = _read(Path(threshold_dir) / "summary.json")
    robustness = _read(Path(robustness_dir) / "summary.json")
    if not robustness.get("complete"):
        raise ValueError("Complete the robustness study before rendering final validation media")
    if not robustness.get("complete", True):
        raise ValueError("Navigation validation figures require a completed robustness study")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({**theme.MATPLOTLIB, "font.size": 8.5}):
        figures = {
            "fov_los": _fov_los(diagram, output / "report_fov_los_3d.png"),
            "threshold": _threshold(threshold, output / "report_threshold_detection.png"),
            "robustness": _robustness(robustness, output / "report_navigation_robustness.png"),
        }
    figures["html"] = _offline_html(diagram, threshold, robustness, figures,
                                   output / "navigation_validation.html")
    return figures
