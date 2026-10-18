"""Static figures and animations from completed navigation results."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .. import api as ls
from . import theme
from .scene_geometry import footprint_outline


def _color(name, i):
    return theme.SCENARIO_COLORS.get(name, f"C{i}")


def draw_scene(ax, scene, nav):
    for o in scene.objects:
        poly = footprint_outline(o)
        if poly is not None:
            ax.fill(
                poly[:, 0], poly[:, 1], color=theme.color("0.35"), alpha=0.85, zorder=3
            )
            c = poly[:-1].mean(0)
            ax.text(
                c[0],
                c[1],
                o.label,
                fontsize=6,
                ha="center",
                va="center",
                color=theme.color("w"),
                zorder=4,
                clip_on=True,
            )
    s, g = (nav["robot"]["start"], nav["robot"]["goal"])
    ax.plot(*s, "o", ms=6, zorder=5, color=theme.color("k"))
    ax.plot(*g, "*", ms=12, zorder=5, color=theme.color("k"))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


@theme.mpl_style
def plot_trajectories(results, nav, path):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(
        len(results),
        1,
        figsize=(10, 3.6 * len(results)),
        squeeze=False,
        layout="constrained",
    )
    im = None
    mapped_axes = []
    for ax, ref in zip(axs[:, 0], results):
        g = ref["grid"]
        if ref.get("grid_available", True):
            im = ax.imshow(
                g.prob(),
                origin="lower",
                extent=[g.x0, g.x1, g.y0, g.y1],
                cmap=theme.colormap("occupancy"),
                vmin=0,
                vmax=1,
                alpha=0.55,
            )
            mapped_axes.append(ax)
        draw_scene(ax, ref["scene"], nav)
        for i, r in enumerate(results):
            own = r is ref
            ax.plot(
                r["log"][:, 1],
                r["log"][:, 2],
                "-",
                lw=2.2 if own else 0.8,
                alpha=1 if own else 0.6,
                color=_color(r["name"], i),
                label=f"{r['name']} ({r['status']}, {r['log'][-1, 0]:.0f} s)",
            )
        ax.set_xlim(-5, 32)
        ax.set_ylim(-7, 8)
        ax.legend(loc="upper left", fontsize=7)
        title = (
            f"Occupancy map built in '{ref['name']}' (green = free, red = occupied, yellow = unknown)"
            if ref.get("grid_available", True)
            else f"Trajectory in '{ref['name']}' (occupancy map was not saved)"
        )
        ax.set_title(title, fontsize=9)
    if im is not None:
        fig.colorbar(im, ax=mapped_axes, shrink=0.6, label="P(occupied)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


@theme.mpl_style
def plot_clearance(results, nav, path):
    import matplotlib.pyplot as plt

    rr = nav["robot"]["radius_m"]
    n = len(results)
    fig, axs = plt.subplots(
        n + 1, 1, figsize=(9, 2.3 * (n + 1)), sharex=True, squeeze=False
    )
    for i, r in enumerate(results):
        lg = r["log"]
        axs[n, 0].plot(
            lg[:, 0],
            100 * (lg[:, 7] - lg[:, 6]),
            ".-",
            lw=0.6,
            ms=3,
            color=_color(r["name"], i),
            label=r["name"],
        )
    axs[n, 0].axhline(0, color=theme.color("k"), lw=0.8)
    axs[n, 0].set_ylim(-8, 8)
    axs[n, 0].set_ylabel("$\\hat d - d$ [cm]")
    axs[n, 0].set_title(
        "perception error of the clearance (off-scale = nearest obstacle not detected)",
        fontsize=9,
    )
    axs[n, 0].legend(fontsize=7, ncol=3)
    for i, (ax, r) in enumerate(zip(axs[:n, 0], results)):
        lg = r["log"]
        ax.plot(
            lg[:, 0],
            lg[:, 6],
            "-",
            lw=1.6,
            label="true clearance $d$",
            color=theme.color("k"),
        )
        pc = np.where(np.isfinite(lg[:, 7]), lg[:, 7], np.nan)
        ax.plot(
            lg[:, 0],
            pc,
            ".",
            color=_color(r["name"], i),
            ms=3,
            label="perceived $\\hat d$",
        )
        ax.axhline(
            rr,
            color=theme.color("r"),
            ls="--",
            lw=1,
            label="collision ($d$ = robot radius)",
        )
        ax.set_ylim(0, 6)
        ax.set_ylabel("distance [m]")
        ax.set_title(
            f"{r['name']}: alpha = {r['cfg'].extinction_per_m:.3f} 1/m", fontsize=9
        )
        ax.legend(fontsize=7, loc="upper right")
    axs[-1, 0].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@theme.mpl_style
def plot_detection(results, nav, path):
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    cfg0 = results[0]["cfg"]
    alphas = np.logspace(-4, 0, 60)
    for rho, ls_ in ((0.05, ":"), (0.1, "--"), (0.3, "-"), (0.85, "-.")):
        R = [
            ls.max_range_estimate(
                ls.replace(cfg0, air_extinction_per_m=a, rain_mm_per_h=0.0), rho
            )
            for a in alphas
        ]
        a1.loglog(
            alphas, R, ls_, color=theme.color("black"), lw=1.2, label=f"$\\rho$ = {rho}"
        )
    for i, r in enumerate(results):
        a1.axvline(
            r["cfg"].extinction_per_m,
            color=_color(r["name"], i),
            lw=2,
            alpha=0.7,
            label=r["name"],
        )
    a1.set_xlabel("extinction coefficient $\\alpha$ [1/m]")
    a1.set_ylabel("max. detection range $R_{max}$ [m]")
    a1.set_title("FOV radius from the LiDAR equation (normal incidence)")
    a1.grid(True, which="both", alpha=0.3)
    a1.legend(fontsize=7)
    labels = [k for k in results[0]["first_det"]]
    xs = np.arange(len(labels))
    wd = 0.8 / len(results)
    for i, r in enumerate(results):
        vals = [r["first_det"][k] for k in labels]
        a2.bar(
            xs + i * wd,
            np.nan_to_num(vals, nan=0.0),
            wd,
            color=_color(r["name"], i),
            label=r["name"],
        )
    a2.set_xticks(xs + 0.4 - wd / 2)
    a2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    a2.set_ylabel("distance at first detection [m]")
    a2.set_title("Range at which each obstacle is first perceived (0 = never)")
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@theme.mpl_style
def animate(res, nav, path, every, fps):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from ..navigation.storage import require_frames

    require_frames(res)
    lg, fr, rb = (res["log"], res["frames"], nav["robot"])
    idx = list(range(0, len(fr), every))
    rmax = ls.max_range_estimate(res["cfg"], 0.1)
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    def draw(j):
        k = idx[j]
        ax.clear()
        ax2.clear()
        draw_scene(ax, res["scene"], nav)
        x, y, th = lg[k, 1:4]
        f = fr[k]
        ax.plot(
            f["gnd"][:, 0],
            f["gnd"][:, 1],
            ".",
            color=theme.color("#9ecae1"),
            ms=1,
            zorder=1,
        )
        ax.plot(
            f["obs"][:, 0],
            f["obs"][:, 1],
            ".",
            color=theme.color("#e6550d"),
            ms=2,
            zorder=6,
        )
        ends = np.stack(
            [x + f["L"] * np.cos(res["H"]), y + f["L"] * np.sin(res["H"])], 1
        )
        for e in ends:
            ax.plot(
                [x, e[0]],
                [y, e[1]],
                "-",
                color=theme.color("#31a354"),
                lw=0.5,
                alpha=0.5,
                zorder=2,
            )
        ax.add_patch(
            plt.Circle(
                (x, y),
                min(rmax, 60),
                fill=False,
                ls=":",
                color=theme.color("purple"),
                zorder=2,
            )
        )
        ax.add_patch(
            plt.Circle((x, y), rb["radius_m"], color=theme.color("k"), zorder=7)
        )
        ax.arrow(
            x,
            y,
            0.9 * np.cos(th),
            0.9 * np.sin(th),
            width=0.06,
            color=theme.color("y"),
            zorder=8,
        )
        ax.arrow(
            x,
            y,
            1.6 * np.cos(f["psi"]),
            1.6 * np.sin(f["psi"]),
            width=0.03,
            color=theme.color("g"),
            zorder=8,
        )
        ax.plot(
            lg[: k + 1, 1], lg[: k + 1, 2], "-", lw=1, zorder=5, color=theme.color("k")
        )
        ax.set_xlim(-5, 32)
        ax.set_ylim(-9, 9)
        ax.set_title(
            f"{res['name']}   t = {lg[k, 0]:.1f} s   v = {lg[k, 4]:.2f} m/s   returns = {int(lg[k, 8])}  (dotted: R_max for rho = 0.1)",
            fontsize=9,
        )
        ax2.plot(
            lg[: k + 1, 0],
            lg[: k + 1, 6],
            "-",
            label="true $d$",
            color=theme.color("k"),
        )
        ax2.plot(
            lg[: k + 1, 0],
            lg[: k + 1, 7],
            ".",
            color=theme.color("#e6550d"),
            ms=3,
            label="perceived $\\hat d$",
        )
        ax2.axhline(rb["radius_m"], color=theme.color("r"), ls="--", lw=1)
        ax2.set_xlim(0, lg[-1, 0] + 0.1)
        ax2.set_ylim(0, 6)
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("clearance [m]")
        ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=len(idx))
    anim.save(path, writer=PillowWriter(fps=fps))
    draw(int(0.47 * (len(idx) - 1)))
    fig.savefig(str(Path(path).with_name(Path(path).stem + "_snapshot.png")), dpi=150)
    plt.close(fig)
