"""Local browser transport and display-frame serialization; no simulation loop."""
from __future__ import annotations
import base64
import io
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import numpy as np
from .. import api as ls
from . import theme
from .first_person import FirstPerson
from .live import LiveView
from .scene_geometry import footprint_outline


def _png_uri(img):
    import matplotlib.image as mpimg

    buf = io.BytesIO()
    mpimg.imsave(buf, img, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _thin(a, n):
    return a[:: max(1, int(np.ceil(len(a) / n)))]


class Store:
    def __init__(self, port=8765, open_browser=True):
        self.lock = threading.Lock()
        self.run = 0
        self.hist, self.xs, self.ys = [], [], []
        self.meta = {"run": 0, "n": 0}
        self.cam = None
        self.port, self.open_browser, self.server = port, open_browser, None
        self.thread = None

    # ---- server ----
    def ensure_server(self):
        if self.server is not None:
            return
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), make_handler(self))
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        url = f"http://127.0.0.1:{self.port}/"
        print(f"Plotly viewer: {url}")
        if self.open_browser:
            webbrowser.open(url)

    # ---- simulation side ----
    def reset(self, st, view):
        nav = self.nav = view.nav
        self.cam = FirstPerson(st["scene"], nav["mount_height_m"])
        self.alpha = st["cfg"].extinction_per_m
        self.rmax = float(ls.max_range_estimate(st["cfg"], 0.1))
        objs = []
        for o, lab in zip(st["scene"].objects, st["scene"].labels):
            poly = footprint_outline(o)
            if poly is not None:
                objs.append(
                    {
                        "label": lab,
                        "x": np.round(poly[:, 0], 3).tolist(),
                        "y": np.round(poly[:, 1], 3).tolist(),
                    }
                )
        g, rb = st["grid"], nav["robot"]
        with self.lock:
            self.run += 1
            self.hist, self.xs, self.ys = [], [], []
            self.meta = {
                "run": self.run,
                "n": 0,
                "name": view.names[view.current],
                "alpha": self.alpha,
                "rmax": self.rmax,
                "objects": objs,
                "extent": [g.x0, g.x1, g.y0, g.y1],
                "res": g.res,
                "H": np.round(st["H"], 5).tolist(),
                "cam_w": self.cam.w,
                "cam_h": self.cam.h,
                "hfov": float(self.cam.hfov),
                "radius": rb["radius_m"],
                "r_inf": rb["radius_m"] + nav["planner"]["safety_margin_m"],
                "start": list(rb["start"]),
                "goal": list(rb["goal"]),
            }

    def push(self, st, view):
        if st["k"] == 0 or self.cam is None:
            self.reset(st, view)
        t, x, y, th, v, w, d_true, d_perc = st["log"][-1][:8]
        pc = st["pc"]
        img, depth = self.cam.render(x, y, th, self.alpha)
        u, r, rr = self.cam.project(pc.xyz_world, x, y, th, depth)
        if len(u) > 1500:
            sel = np.random.default_rng(0).choice(len(u), 1500, replace=False)
            u, r, rr = u[sel], r[sel], rr[sel]
        obs, gnd = _thin(st["obs"], 2000), _thin(st["gnd"], 1500)
        frame = dict(
            img=_png_uri(img),
            cam=(np.round(u, 1), np.round(r, 1), np.round(rr, 2)),
            obs=np.round(obs, 2),
            gnd=np.round(gnd, 2),
            L=None if st["L"] is None else np.round(st["L"], 2),
            psi=float(st["psi"]),
            x=float(x),
            y=float(y),
            th=float(th),
            t=float(t),
            v=float(v),
            w=float(w),
            dtrue=float(d_true),
            dperc=float(d_perc),
            n=int(pc.n),
            n_obs=int(len(st["obs"])),
            status=st["status"],
            grid=(st["grid"].prob() * 255).astype(np.uint8),
        )
        with self.lock:
            self.hist.append(frame)
            self.xs.append(float(x))
            self.ys.append(float(y))
            self.meta["n"] = len(self.hist)

    # ---- server side ----
    def get_meta(self):
        with self.lock:
            return dict(self.meta)

    def get_frame(self, i):
        with self.lock:
            if not self.hist:
                return None
            i = len(self.hist) - 1 if i < 0 else min(i, len(self.hist) - 1)
            f = self.hist[i]
            xs, ys = self.xs[: i + 1], self.ys[: i + 1]
        u, r, rr = f["cam"]
        return {
            "i": i,
            "img": f["img"],
            "cam": {"u": u.tolist(), "v": r.tolist(), "r": rr.tolist()},
            "obs": {"x": f["obs"][:, 0].tolist(), "y": f["obs"][:, 1].tolist()},
            "gnd": {"x": f["gnd"][:, 0].tolist(), "y": f["gnd"][:, 1].tolist()},
            "L": None if f["L"] is None else f["L"].tolist(),
            "psi": f["psi"],
            "x": f["x"],
            "y": f["y"],
            "th": f["th"],
            "t": f["t"],
            "v": f["v"],
            "w": f["w"],
            "dtrue": f["dtrue"] if np.isfinite(f["dtrue"]) else None,
            "dperc": f["dperc"] if np.isfinite(f["dperc"]) else None,
            "n": f["n"],
            "n_obs": f["n_obs"],
            "status": f["status"],
            "grid": np.round(f["grid"] / 255.0, 2).tolist(),
            "trail": {"x": xs, "y": ys},
        }

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread is not None:
            self.thread.join(timeout=1)
            self.thread = None


def make_handler(store):
    """Bind each local server to its own frame store, without process globals."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype):
            if isinstance(body, str):
                body = body.encode("utf-8")
            try:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                page = theme.asset("templates/live.html")
                page = page.replace("__STYLE__", theme.stylesheet("live.css"))
                settings = json.dumps(theme.web_settings(), ensure_ascii=False).replace(
                    "</", "<\\/"
                )
                self._send(
                    page.replace("__THEME__", settings), "text/html; charset=utf-8"
                )
            elif url.path == "/assets/live.js":
                self._send(
                    theme.asset("static/live.js"), "text/javascript; charset=utf-8"
                )
            elif url.path == "/plotly.js":
                from plotly.offline import get_plotlyjs

                self._send(get_plotlyjs(), "text/javascript; charset=utf-8")
            elif url.path == "/meta":
                self._send(
                    json.dumps(store.get_meta(), allow_nan=False), "application/json"
                )
            elif url.path == "/frame":
                try:
                    index = int(parse_qs(url.query).get("i", ["-1"])[0])
                except ValueError:
                    self.send_error(400, "frame index must be an integer")
                    return
                self._send(
                    json.dumps(store.get_frame(index), allow_nan=False),
                    "application/json",
                )
            else:
                self.send_error(404)

    return Handler


class BrowserLiveView(LiveView):
    """The native live view with an independent browser replay surface."""

    def __init__(self, nav, speed=1.0, *, port=8765, open_browser=True):
        super().__init__(nav, speed)
        self.store = Store(port=port, open_browser=open_browser)
        self.store.ensure_server()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.store.port}/"

    def on_step(self, state):
        self.store.push(state, self)
        return super().on_step(state)

    def close(self):
        self.store.close()
        super().close()
