"""Callback-driven navigation views; simulation orchestration belongs to the CLI."""
from __future__ import annotations
import time
import numpy as np
from .. import api as ls
from . import theme
from .first_person import FirstPerson
from .navigation_figures import draw_scene


class LiveView:
    def __init__(self, nav, speed):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        self.plt, self.LC = (plt, LineCollection)
        self.nav, self.speed = (nav, speed)
        self.names = [s["name"] for s in nav["scenarios"]]
        self.paused = self.follow = False
        self.show_lidar = self.show_map = True
        self.request = None
        self.fig = None
        self.current = 0

    @theme.mpl_style
    def setup(self, res0):
        plt = self.plt
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig = plt.figure(figsize=(15, 8.2))
            self.fig.canvas.mpl_connect("key_press_event", self.on_key)
            self.fig.canvas.mpl_connect(
                "close_event", lambda e: setattr(self, "request", "quit")
            )
        fig = self.fig
        fig.clf()
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[3.2, 1],
            width_ratios=[1.15, 1],
            hspace=0.28,
            wspace=0.12,
            left=0.05,
            right=0.98,
            top=0.93,
            bottom=0.07,
        )
        self.ax_top, self.ax_cam = (
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
        )
        self.ax_d = fig.add_subplot(gs[1, :])
        nav, scene, cfg, grid = (self.nav, res0["scene"], res0["cfg"], res0["grid"])
        rb = nav["robot"]
        self.alpha = cfg.extinction_per_m
        self.rmax = ls.max_range_estimate(cfg, 0.1)
        self.cam = FirstPerson(scene, nav["mount_height_m"])
        a = self.ax_top
        self.im_map = a.imshow(
            grid.prob(),
            origin="lower",
            extent=[grid.x0, grid.x1, grid.y0, grid.y1],
            cmap=theme.colormap("occupancy"),
            vmin=0,
            vmax=1,
            alpha=0.5,
            zorder=0,
        )
        draw_scene(a, scene, nav)
        (self.sc_gnd,) = a.plot(
            [], [], ".", color=theme.color("#6baed6"), ms=1, zorder=1
        )
        (self.sc_obs,) = a.plot(
            [], [], ".", color=theme.color("#e6550d"), ms=2.5, zorder=6
        )
        self.fan = self.LC(
            [], colors=theme.color("#31a354"), linewidths=0.6, alpha=0.6, zorder=2
        )
        a.add_collection(self.fan)
        self.fov_c = plt.Circle(
            (0, 0), 1, fill=False, ls=":", color=theme.color("purple"), lw=1.2, zorder=2
        )
        a.add_patch(self.fov_c)
        (self.cam_fov,) = a.plot(
            [], [], "-", color=theme.color("gold"), lw=1.2, zorder=7
        )
        (self.trail,) = a.plot([], [], "-", lw=1.2, zorder=5, color=theme.color("k"))
        self.body = plt.Circle((0, 0), rb["radius_m"], color=theme.color("k"), zorder=8)
        a.add_patch(self.body)
        (self.head,) = a.plot(
            [], [], "-", color=theme.color("yellow"), lw=2.5, zorder=9
        )
        (self.goal_dir,) = a.plot(
            [], [], "-", color=theme.color("#00c000"), lw=2, zorder=9
        )
        self.hud = a.text(
            0.01,
            0.99,
            "",
            transform=a.transAxes,
            va="top",
            fontsize=8,
            family="monospace",
            bbox=dict(fc=theme.color("w"), alpha=0.8, lw=0),
            zorder=10,
        )
        self.set_view(rb["start"])
        c = self.ax_cam
        img, _ = self.cam.render(
            rb["start"][0],
            rb["start"][1],
            np.radians(rb["start_heading_deg"]),
            self.alpha,
        )
        self.im_cam = c.imshow(
            img, extent=[0, self.cam.w, self.cam.h, 0], interpolation="bilinear"
        )
        self.sc_cam = c.scatter(
            [],
            [],
            s=4,
            c=[],
            cmap=theme.colormap("camera_returns"),
            vmin=0,
            vmax=min(self.rmax, 30),
            zorder=3,
        )
        c.set_xlim(0, self.cam.w)
        c.set_ylim(self.cam.h, 0)
        c.set_xticks([])
        c.set_yticks([])
        self.cam_txt = c.text(
            0.01,
            0.02,
            "",
            transform=c.transAxes,
            color=theme.color("w"),
            fontsize=8,
            bbox=dict(fc=theme.color("k"), alpha=0.5, lw=0),
        )
        cb = fig.colorbar(
            self.sc_cam, ax=c, orientation="horizontal", fraction=0.05, pad=0.02
        )
        cb.set_label("LiDAR return range [m]", fontsize=8)
        c.set_title("first-person view (render + live LiDAR returns)", fontsize=10)
        d = self.ax_d
        (self.l_true,) = d.plot(
            [], [], "-", lw=1.5, label="true distance d", color=theme.color("k")
        )
        (self.l_perc,) = d.plot(
            [],
            [],
            ".",
            color=theme.color("#e6550d"),
            ms=3,
            label="perceived d-hat (LiDAR)",
        )
        (self.l_v,) = d.plot(
            [], [], "-", color=theme.color("#3182bd"), lw=1, label="speed v [m/s]"
        )
        d.axhline(
            rb["radius_m"], color=theme.color("r"), ls="--", lw=1, label="collision"
        )
        d.set_xlim(0, rb["max_steps"] * rb["dt"] * 0.35)
        d.set_ylim(0, 6)
        d.set_xlabel("time [s]")
        d.legend(fontsize=7, loc="upper right", ncol=4)
        fig.suptitle(
            f"scenario '{self.names[self.current]}'   alpha = {self.alpha:.3g} 1/m   R_max(rho=0.1) = {self.rmax:.1f} m        keys: space pause | {' '.join((str(i + 1) for i in range(len(self.names))))} scenario ({', '.join(self.names)}) | r restart | f follow | l lidar | m map | q quit",
            fontsize=9,
        )

    def set_view(self, xy):
        if self.follow:
            self.ax_top.set_xlim(xy[0] - 9, xy[0] + 9)
            self.ax_top.set_ylim(xy[1] - 6.5, xy[1] + 6.5)
        else:
            self.ax_top.set_xlim(-5, 32)
            self.ax_top.set_ylim(-9, 9)

    def on_key(self, e):
        k = e.key
        if k == " ":
            self.paused = not self.paused
        elif k == "q":
            self.request = "quit"
        elif k == "r":
            self.request = self.current
        elif k and k.isdigit() and (1 <= int(k) <= len(self.names)):
            self.request = int(k) - 1
        elif k == "f":
            self.follow = not self.follow
        elif k == "l":
            self.show_lidar = not self.show_lidar
        elif k == "m":
            self.show_map = not self.show_map

    def on_step(self, st):
        t0 = time.time()
        if st["k"] == 0:
            self.setup(st)
        lg = np.array(st["log"])
        t, x, y, th, v, w, dtrue, dperc = lg[-1, :8]
        rb = self.nav["robot"]
        self.im_map.set_data(st["grid"].prob())
        self.im_map.set_visible(self.show_map)
        self.sc_obs.set_data(st["obs"][:, 0], st["obs"][:, 1])
        self.sc_gnd.set_data(st["gnd"][::3, 0], st["gnd"][::3, 1])
        if st["L"] is not None:
            H, L = (st["H"], st["L"])
            self.fan.set_segments(
                [[(x, y), (x + l * np.cos(h), y + l * np.sin(h))] for h, l in zip(H, L)]
            )
        self.fov_c.center, self.fov_c.radius = ((x, y), min(self.rmax, 60))
        hf = self.cam.hfov / 2
        r = 6.0
        self.cam_fov.set_data(
            [x + r * np.cos(th + hf), x, x + r * np.cos(th - hf)],
            [y + r * np.sin(th + hf), y, y + r * np.sin(th - hf)],
        )
        self.trail.set_data(lg[:, 1], lg[:, 2])
        self.body.center = (x, y)
        self.head.set_data([x, x + 0.9 * np.cos(th)], [y, y + 0.9 * np.sin(th)])
        self.goal_dir.set_data(
            [x, x + 1.5 * np.cos(st["psi"])], [y, y + 1.5 * np.sin(st["psi"])]
        )
        self.set_view((x, y))
        self.hud.set_text(
            f"t = {t:5.1f} s   v = {v:4.2f} m/s   omega = {np.degrees(w):6.1f} deg/s\ntrue clearance      {dtrue - rb['radius_m']:5.2f} m (surface gap)\nperceived clearance {dperc - rb['radius_m']:5.2f} m\nreturns {st['pc'].n:5d}  obstacle pts {len(st['obs']):5d}   {('PAUSED' if self.paused else st['status'])}"
        )
        img, depth = self.cam.render(x, y, th, self.alpha)
        self.im_cam.set_data(img)
        if self.show_lidar:
            u, vv, rr = self.cam.project(st["pc"].xyz_world, x, y, th, depth)
            self.sc_cam.set_offsets(np.column_stack([u, vv]))
            self.sc_cam.set_array(rr)
            self.sc_cam.set_sizes(np.clip(40.0 / np.maximum(rr, 1), 1, 12))
        else:
            self.sc_cam.set_offsets(np.zeros((0, 2)))
        self.cam_txt.set_text(
            f"nearest perceived obstacle: {dperc:.2f} m   visibility ~ {min(3.9 / self.alpha, 9999):.0f} m"
        )
        self.l_true.set_data(lg[:, 0], lg[:, 6])
        self.l_perc.set_data(
            lg[:, 0], np.where(np.isfinite(lg[:, 7]), lg[:, 7], np.nan)
        )
        self.l_v.set_data(lg[:, 0], lg[:, 4])
        if t > self.ax_d.get_xlim()[1] - 1:
            self.ax_d.set_xlim(0, t + 10)
        wait = (rb["dt"] / self.speed if self.speed > 0 else 0.0) - (time.time() - t0)
        self.plt.pause(max(wait, 0.001))
        while self.paused and self.request is None:
            self.plt.pause(0.05)
        return self.request is None

    def close(self):
        """Close this view without touching other open figures."""
        if self.fig is not None:
            self.plt.close(self.fig)


def create_live_view(
    nav, speed=1.0, *, backend="matplotlib", port=8765, open_browser=True
):
    """Create a callback-driven view; the caller owns simulation and restart loops."""
    if backend == "matplotlib":
        return LiveView(nav, speed)
    if backend == "browser":
        from .browser import BrowserLiveView

        return BrowserLiveView(nav, speed, port=port, open_browser=open_browser)
    raise ValueError("backend must be 'matplotlib' or 'browser'")


def iter_replay_states(result):
    """Yield stored callback states without invoking the simulator or planner."""
    import copy

    history = result.get("history", [])
    if not history:
        raise ValueError(
            "this saved navigation run has no full replay frames; render its summary plots instead"
        )
    grid = copy.deepcopy(result["grid"])
    log = np.asarray(result["log"])
    for i, frame in enumerate(history):
        grid.l[...] = frame["grid_l"]
        yield {
            "k": frame.get("k", i),
            "cfg": result["cfg"],
            "scene": result["scene"],
            "grid": grid,
            "H": result["H"],
            "log": log[: frame.get("k", i) + 1],
            "pc": frame["pc"],
            "obs": frame["obs"],
            "gnd": frame["gnd"],
            "L": frame["L"],
            "psi": frame["psi"],
            "status": frame["status"],
        }
