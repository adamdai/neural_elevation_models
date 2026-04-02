from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import webbrowser

import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import torch
import tyro

from nemo import Nemo
from nemo.particle_sim import ParticleSimConfig, particle_config_payload, sample_height_field_grid, simulate_particle


@dataclass
class VisualizeOptimizationArgs:
    checkpoint_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    host: str = "127.0.0.1"
    port: int = 8057
    grid_resolution_x: int = 180
    grid_resolution_y: int = 180
    batch_size: int = 65536
    surface_opacity: float = 0.95
    open_browser: bool = True
    particle_grid_size: int = 10
    dt: float = 0.02
    max_steps: int = 1500
    gravity: float = 9.81
    spawn_height: float = 0.1
    air_damping: float = 0.05
    surface_damping: float = 1.5
    static_friction_slope: float = 0.015
    settle_speed: float = 0.02
    settle_slope: float = 0.01
    downhill_scale: float = 1.0


class OptimizationVizApp:
    def __init__(self, args: VisualizeOptimizationArgs) -> None:
        self.args = args
        self.checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
        self.nemo = Nemo.load_checkpoint(self.checkpoint_path, map_location=args.device).to(args.device)
        self.config = ParticleSimConfig(
            dt=args.dt,
            max_steps=args.max_steps,
            gravity=args.gravity,
            spawn_height=args.spawn_height,
            air_damping=args.air_damping,
            surface_damping=args.surface_damping,
            static_friction_slope=args.static_friction_slope,
            settle_speed=args.settle_speed,
            settle_slope=args.settle_slope,
            downhill_scale=args.downhill_scale,
        )
        self.surface_x, self.surface_y, self.surface_z = sample_height_field_grid(
            self.nemo,
            resolution_x=args.grid_resolution_x,
            resolution_y=args.grid_resolution_y,
            batch_size=args.batch_size,
        )
        self.bounds = self.nemo.field.bounds
        self.spawn_points = self._build_spawn_grid(args.particle_grid_size)
        self.initial_results = [simulate_particle(self.nemo, xy, config=self.config) for xy in self.spawn_points]
        self._lock = threading.Lock()

    def _build_spawn_grid(self, grid_size: int) -> list[tuple[float, float]]:
        grid_size = max(int(grid_size), 1)
        x_min, x_max = float(self.bounds[0][0]), float(self.bounds[0][1])
        y_min, y_max = float(self.bounds[1][0]), float(self.bounds[1][1])
        x_margin = 0.05 * max(x_max - x_min, 1e-6)
        y_margin = 0.05 * max(y_max - y_min, 1e-6)
        x_axis = torch.linspace(x_min + x_margin, x_max - x_margin, grid_size, dtype=torch.float32).tolist()
        y_axis = torch.linspace(y_min + y_margin, y_max - y_margin, grid_size, dtype=torch.float32).tolist()
        return [(float(x), float(y)) for y in y_axis for x in x_axis]

    def html(self) -> str:
        figure = go.Figure(
            data=[
                go.Surface(
                    x=self.surface_x,
                    y=self.surface_y,
                    z=self.surface_z,
                    colorscale="Viridis",
                    opacity=float(self.args.surface_opacity),
                    showscale=True,
                    hovertemplate="x=%{{x:.4f}}<br>y=%{{y:.4f}}<br>z=%{{z:.4f}}<extra>terrain</extra>",
                )
            ]
        )
        figure.update_layout(
            title=f"NeMO Optimization Viewer: {self.checkpoint_path.name}",
            template="plotly_white",
            margin=dict(l=0, r=0, t=48, b=0),
            scene=dict(
                aspectmode="data",
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
                camera=dict(eye=dict(x=1.35, y=1.35, z=0.85)),
            ),
        )

        figure_json = json.dumps(figure.to_plotly_json(), cls=PlotlyJSONEncoder)
        bounds_json = json.dumps(
            {
                "x": [float(self.bounds[0][0]), float(self.bounds[0][1])],
                "y": [float(self.bounds[1][0]), float(self.bounds[1][1])],
            }
        )
        config_json = json.dumps(particle_config_payload(self.config))
        initial_results_json = json.dumps([result.to_payload() for result in self.initial_results])

        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>NeMO Optimization Viewer</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
      body {{
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #f3f4ef;
        color: #171717;
      }}
      .layout {{
        display: grid;
        grid-template-columns: 320px minmax(0, 1fr);
        min-height: 100vh;
      }}
      .panel {{
        padding: 20px 18px;
        border-right: 1px solid #d6d4cb;
        background:
          radial-gradient(circle at top, rgba(65, 120, 84, 0.15), transparent 36%),
          linear-gradient(180deg, #f8f8f3 0%, #ece9df 100%);
      }}
      .panel h1 {{
        margin: 0 0 10px;
        font-size: 1.1rem;
      }}
      .panel p, .panel li, .panel code {{
        font-size: 0.93rem;
        line-height: 1.45;
      }}
      .panel ul {{
        padding-left: 18px;
      }}
      .meta {{
        margin-top: 16px;
        padding: 12px;
        border: 1px solid #d6d4cb;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.7);
      }}
      .status {{
        margin-top: 16px;
        padding: 12px;
        border-radius: 12px;
        background: #171717;
        color: #f7f7f4;
        min-height: 66px;
      }}
      button {{
        margin-top: 12px;
        width: 100%;
        padding: 10px 12px;
        border: 0;
        border-radius: 999px;
        background: #2d6a4f;
        color: white;
        cursor: pointer;
      }}
      #plot {{
        width: 100%;
        height: 100vh;
      }}
      @media (max-width: 960px) {{
        .layout {{
          grid-template-columns: 1fr;
        }}
        .panel {{
          border-right: 0;
          border-bottom: 1px solid #d6d4cb;
        }}
        #plot {{
          height: 72vh;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="layout">
      <aside class="panel">
        <h1>NeMO Optimization Viewer</h1>
        <p>The viewer starts by spawning a grid of particles across the terrain. Each trajectory is computed in Python using <code>nemo.h(...)</code> and <code>nemo.grad(...)</code> at each simulation step, and the path markers are colored by time.</p>
        <ul>
          <li>Checkpoint: <code>{self.checkpoint_path}</code></li>
          <li>Bounds x: <code>{self.bounds[0][0]:.4f}</code> to <code>{self.bounds[0][1]:.4f}</code></li>
          <li>Bounds y: <code>{self.bounds[1][0]:.4f}</code> to <code>{self.bounds[1][1]:.4f}</code></li>
          <li>Initial particles: <code>{len(self.initial_results)}</code></li>
        </ul>
        <div class="meta">
          <strong>Particle config</strong>
          <pre id="config-box" style="white-space: pre-wrap; margin: 8px 0 0;"></pre>
        </div>
        <div id="status" class="status">Rendering initial particle grid across the terrain.</div>
        <button id="clear-btn" type="button">Clear Particle Traces</button>
      </aside>
      <main>
        <div id="plot"></div>
      </main>
    </div>
    <script>
      const figure = {figure_json};
      const particleConfig = {config_json};
      const initialResults = {initial_results_json};
      const plotEl = document.getElementById("plot");
      const statusEl = document.getElementById("status");
      const configBox = document.getElementById("config-box");
      const clearBtn = document.getElementById("clear-btn");
      let traceCounter = 0;

      configBox.textContent = JSON.stringify(particleConfig, null, 2);

      Plotly.newPlot(plotEl, figure.data, figure.layout, {{responsive: true, displaylogo: false}});

      function addTrajectory(result) {{
        const xyz = result.trajectory_xyz;
        const denom = Math.max(xyz.length - 1, 1);
        const times = xyz.map((_, idx) => idx / denom);
        traceCounter += 1;
        return Plotly.addTraces(plotEl, [
          {{
            type: "scatter3d",
            mode: "lines",
            x: xyz.map((row) => row[0]),
            y: xyz.map((row) => row[1]),
            z: xyz.map((row) => row[2]),
            line: {{color: "rgba(25, 25, 25, 0.18)", width: 3}},
            name: "particle " + traceCounter,
            hovertemplate: "x=%{{x:.4f}}<br>y=%{{y:.4f}}<br>z=%{{z:.4f}}<extra>trajectory</extra>",
          }},
          {{
            type: "scatter3d",
            mode: "markers",
            x: xyz.map((row) => row[0]),
            y: xyz.map((row) => row[1]),
            z: xyz.map((row) => row[2]),
            marker: {{
              size: 2.8,
              color: times,
              cmin: 0,
              cmax: 1,
              colorscale: "Turbo",
              opacity: 0.9,
              showscale: traceCounter === 1,
              colorbar: traceCounter === 1 ? {{
                title: "time",
                len: 0.4,
                x: 1.02,
                y: 0.5
              }} : undefined,
            }},
            name: "time " + traceCounter,
            hovertemplate: "t=%{{marker.color:.2f}}<br>x=%{{x:.4f}}<br>y=%{{y:.4f}}<br>z=%{{z:.4f}}<extra>time</extra>",
            showlegend: false,
          }},
          {{
            type: "scatter3d",
            mode: "markers",
            x: [xyz[0][0]],
            y: [xyz[0][1]],
            z: [xyz[0][2]],
            marker: {{color: "#0f172a", size: 3, symbol: "circle"}},
            name: "spawn " + traceCounter,
            hovertemplate: "spawn<extra></extra>",
            showlegend: false,
          }},
          {{
            type: "scatter3d",
            mode: "markers",
            x: [xyz[xyz.length - 1][0]],
            y: [xyz[xyz.length - 1][1]],
            z: [xyz[xyz.length - 1][2]],
            marker: {{color: result.status === "rolled_off" ? "#b91c1c" : "#14532d", size: 4, symbol: result.status === "rolled_off" ? "x" : "diamond"}},
            name: "final " + traceCounter,
            hovertemplate: "final<extra></extra>",
            showlegend: false,
          }},
        ]);
      }}

      Promise.all(initialResults.map((result) => addTrajectory(result))).then(() => {{
        const statusCounts = initialResults.reduce((acc, result) => {{
          acc[result.status] = (acc[result.status] || 0) + 1;
          return acc;
        }}, {{}});
        const summary = Object.entries(statusCounts)
          .map(([key, value]) => key + ": " + value)
          .join(", ");
        statusEl.textContent = "Rendered " + initialResults.length + " particles. Status counts: " + summary;
      }});

      clearBtn.addEventListener("click", async () => {{
        const total = plotEl.data.length;
        if (total > 1) {{
          const idx = [];
          for (let i = 1; i < total; i += 1) {{
            idx.push(i);
          }}
          await Plotly.deleteTraces(plotEl, idx);
        }}
        statusEl.textContent = "Cleared particle traces.";
      }});
    </script>
  </body>
</html>"""

    def simulate(self, x: float, y: float) -> dict[str, object]:
        with self._lock:
            result = simulate_particle(self.nemo, (x, y), config=self.config)
        return result.to_payload()


def _make_handler(app: OptimizationVizApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in ("/", "/index.html"):
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            html = app.html().encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def do_POST(self) -> None:
            if self.path != "/simulate":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
                x = float(payload["x"])
                y = float(payload["y"])
                result = app.simulate(x, y)
            except Exception as exc:
                message = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
                return

            encoded = json.dumps(result).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return Handler


def _bind_server(
    app: OptimizationVizApp,
    *,
    host: str,
    preferred_port: int,
    max_tries: int = 32,
) -> tuple[ThreadingHTTPServer, int]:
    last_error: OSError | None = None
    for port in range(int(preferred_port), int(preferred_port) + int(max_tries)):
        try:
            server = ThreadingHTTPServer((host, port), _make_handler(app))
            return server, port
        except OSError as exc:
            last_error = exc
            if exc.errno != 98:
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to bind viewer server.")


def main(args: VisualizeOptimizationArgs) -> None:
    app = OptimizationVizApp(args)
    server, port = _bind_server(app, host=args.host, preferred_port=args.port)
    url = f"http://{args.host}:{port}/"
    print(f"Loaded checkpoint: {app.checkpoint_path}", flush=True)
    print(f"Serving viewer at {url}", flush=True)
    print(
        "Checkpoint format note: the file must be a full NeMO checkpoint saved by Nemo.save_checkpoint().",
        flush=True,
    )
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main(tyro.cli(VisualizeOptimizationArgs))
