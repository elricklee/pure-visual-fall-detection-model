from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an HTML report for a fall alert system run.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Directory containing run_summary.json and events.jsonl.")
    parser.add_argument("--output", type=Path, default=None, help="HTML output path. Defaults to <run-dir>/report.html.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_events(path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    alerts: list[dict[str, Any]] = []
    clip_paths: dict[str, str] = {}
    if not path.exists():
        return alerts, clip_paths

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") == "event_start" and isinstance(record.get("alert"), dict):
            alerts.append(record["alert"])
        elif record.get("type") == "event_clip_saved":
            event_id = str(record.get("event_id", ""))
            clip_path = str(record.get("clip_path", ""))
            if event_id and clip_path:
                clip_paths[event_id] = clip_path
    return alerts, clip_paths


def _read_frame_traces(path: Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    if not path.exists():
        return traces
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") == "frame_trace":
            traces.append(record)
    return traces


def _as_rel(path_value: str | None, base_dir: Path) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _fmt(value: Any, default: str = "-") -> str:
    if value is None or value == "":
        return default
    return html.escape(str(value))


def _event_card(alert: dict[str, Any], clip_paths: dict[str, str], run_dir: Path) -> str:
    event_id = str(alert.get("event_id", ""))
    snapshot = _as_rel(str(alert.get("snapshot_path", "")), run_dir)
    clip = _as_rel(clip_paths.get(event_id) or str(alert.get("clip_path", "")), run_dir)
    score = alert.get("score")
    score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else _fmt(score)
    snapshot_html = (
        f'<img class="snapshot" src="{html.escape(snapshot)}" alt="snapshot for {html.escape(event_id)}">'
        if snapshot
        else '<div class="missing">No snapshot</div>'
    )
    replay_html = (
        f'<video class="replay" src="{html.escape(clip)}" controls preload="metadata"></video>'
        if clip
        else '<div class="missing">Replay pending</div>'
    )
    return f"""
        <article class="event-card">
          <div class="event-media">{snapshot_html}{replay_html}</div>
          <div class="event-body">
            <h3>{_fmt(event_id)}</h3>
            <dl>
              <div><dt>Track ID</dt><dd>{_fmt(alert.get("track_id"))}</dd></div>
              <div><dt>State</dt><dd>{_fmt(alert.get("state"))}</dd></div>
              <div><dt>Score</dt><dd>{html.escape(score_text)}</dd></div>
              <div><dt>Time</dt><dd>{_fmt(alert.get("t_sec"))} s</dd></div>
              <div><dt>Frame</dt><dd>{_fmt(alert.get("frame_index"))}</dd></div>
            </dl>
            <p class="reason">{_fmt(alert.get("reason"))}</p>
          </div>
        </article>
    """


def _trace_row(trace: dict[str, Any]) -> str:
    detections = trace.get("detections") if isinstance(trace.get("detections"), list) else []
    detection_text = "<br>".join(
        html.escape(
            f"#{det.get('track_id')} {det.get('state')} det={float(det.get('det_conf', 0.0)):.2f} "
            f"score={float(det.get('score', 0.0)):.2f}"
        )
        for det in detections
        if isinstance(det, dict)
    )
    if not detection_text:
        detection_text = "-"
    active_ids = trace.get("active_ids") if isinstance(trace.get("active_ids"), list) else []
    active_text = ", ".join(str(v) for v in active_ids) if active_ids else "-"
    return f"""
      <tr>
        <td>{_fmt(trace.get("frame_index"))}</td>
        <td>{_fmt(trace.get("t_sec"))}</td>
        <td>{_fmt(trace.get("person_count"))}</td>
        <td>{_fmt(active_text)}</td>
        <td>{detection_text}</td>
      </tr>
    """


def _render_html(
    summary: dict[str, Any],
    alerts: list[dict[str, Any]],
    clip_paths: dict[str, str],
    traces: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    live_video = _as_rel(str(artifacts.get("live_video", "")), run_dir)
    events = int(summary.get("events") or len(alerts))
    fps = float(summary.get("fps") or 0.0)
    frames = int(summary.get("frames") or 0)
    duration = float(summary.get("duration_sec") or 0.0)
    event_cards = "\n".join(_event_card(alert, clip_paths, run_dir) for alert in alerts)
    if not event_cards:
        event_cards = '<section class="empty">No fall alert events were recorded in this run.</section>'

    trace_rows = "\n".join(_trace_row(trace) for trace in traces[-60:])
    if not trace_rows:
        trace_rows = '<tr><td colspan="5">No per-frame trace file found.</td></tr>'

    live_html = (
        f'<video class="live-video" src="{html.escape(live_video)}" controls preload="metadata"></video>'
        if live_video
        else '<div class="empty">No annotated live video path found.</div>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fall Alert Run Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1d2430;
      --muted: #5b6676;
      --line: #d9dee7;
      --accent: #bd1f36;
      --ok: #146c43;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, "Microsoft YaHei", sans-serif;
    }}
    header, main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    header {{
      display: grid;
      gap: 16px;
      border-bottom: 1px solid var(--line);
    }}
    h1, h2, h3, p {{
      margin: 0;
    }}
    h1 {{
      font-size: 32px;
      line-height: 1.2;
    }}
    h2 {{
      font-size: 22px;
      margin: 32px 0 16px;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }}
    .metric, .event-card, .empty {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric {{
      padding: 14px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .metric strong {{
      font-size: 22px;
    }}
    .live-video {{
      width: 100%;
      max-height: 640px;
      background: #111;
      border-radius: 8px;
    }}
    .event-list {{
      display: grid;
      gap: 16px;
    }}
    .trace-box {{
      max-height: 560px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .trace-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .trace-table th,
    .trace-table td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      vertical-align: top;
      text-align: left;
    }}
    .trace-table th {{
      position: sticky;
      top: 0;
      background: #f1f4f8;
      z-index: 1;
    }}
    .event-card {{
      display: grid;
      grid-template-columns: minmax(260px, 0.9fr) minmax(260px, 1fr);
      overflow: hidden;
    }}
    .event-media {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      padding: 12px;
      background: #151922;
    }}
    .snapshot, .replay {{
      width: 100%;
      border-radius: 6px;
      background: #000;
    }}
    .event-body {{
      padding: 18px;
      display: grid;
      gap: 14px;
      align-content: start;
    }}
    dl {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 10px;
      margin: 0;
    }}
    dt {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    dd {{
      margin: 4px 0 0;
      font-weight: 700;
    }}
    .reason {{
      color: var(--muted);
      line-height: 1.5;
      word-break: break-word;
    }}
    .empty, .missing {{
      padding: 18px;
      color: var(--muted);
    }}
    .danger {{
      color: var(--accent);
    }}
    @media (max-width: 760px) {{
      header, main {{
        padding: 16px;
      }}
      .event-card {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Fall Alert Run Report</h1>
      <p>{_fmt(summary.get("run_name"))} | {_fmt(summary.get("started_at"))}</p>
    </div>
    <section class="meta-grid">
      <div class="metric"><span>Events</span><strong class="danger">{events}</strong></div>
      <div class="metric"><span>Frames</span><strong>{frames}</strong></div>
      <div class="metric"><span>FPS</span><strong>{fps:.2f}</strong></div>
      <div class="metric"><span>Duration</span><strong>{duration:.2f}s</strong></div>
      <div class="metric"><span>Resolution</span><strong>{_fmt(summary.get("width"))}x{_fmt(summary.get("height"))}</strong></div>
    </section>
  </header>
  <main>
    <section>
      <h2>Annotated Video</h2>
      {live_html}
    </section>
    <section>
      <h2>Alert Events</h2>
      <div class="event-list">{event_cards}</div>
    </section>
    <section>
      <h2>Frame Trace</h2>
      <p class="reason">Per-frame trace is saved to <code>frames.jsonl</code>. The table below shows the latest frames so testers can inspect track_id, det, score, and state without opening raw logs.</p>
      <div class="trace-box">
        <table class="trace-table">
          <thead>
            <tr>
              <th>Frame</th>
              <th>Time</th>
              <th>People</th>
              <th>Active IDs</th>
              <th>Detections</th>
            </tr>
          </thead>
          <tbody>
            {trace_rows}
          </tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""


def generate_report(run_dir: Path, output: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    output_path = output.resolve() if output else run_dir / "report.html"
    summary = _read_json(run_dir / "run_summary.json")
    alerts, clip_paths = _read_events(run_dir / "events.jsonl")
    traces = _read_frame_traces(run_dir / "frames.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_html(summary, alerts, clip_paths, traces, run_dir), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    report_path = generate_report(args.run_dir, args.output)
    print(f"saved: {report_path}")


if __name__ == "__main__":
    main()
