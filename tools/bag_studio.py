#!/usr/bin/env python3
"""本地点云工作台：选 rosbag → 参数 → 出 2D 地图 + 3D 点云。

只在本机 127.0.0.1 上跑，不依赖任何第三方 Web 框架。
底层依次调用 bag_mapping_pipeline.py 与 slope_map_projector.py，
参数在页面上可见可改，默认值与 bag_to_map.py 的固化值一致。

启动：
    /usr/bin/python3 tools/bag_studio.py           # 自动打开浏览器
    /usr/bin/python3 tools/bag_studio.py --no-open # 不自动开浏览器
"""

import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
PYTHON = "/usr/bin/python3"          # 带 numpy / scipy 的解释器
SERVED_SUFFIXES = {".pgm", ".png", ".pcd", ".yaml", ".json", ".md", ".tum"}

JOBS = {}
JOB_LOCK = threading.Lock()


# ------------------------------------------------------------------ 任务执行

def _run(cmd, job):
    job["log"].append("+ " + " ".join(str(c) for c in cmd))
    process = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        job["log"].append(line.rstrip())
        if len(job["log"]) > 2000:
            del job["log"][:500]
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"命令失败（exit {code}）：{cmd[0]}")
    return code


def run_job(job_id, params):
    job = JOBS[job_id]
    try:
        bag = Path(params["bag"]).expanduser()
        requested = Path(params["out"]).expanduser()
        if not bag.is_file():
            raise RuntimeError(f"找不到 bag：{bag}")
        out, note = _resolve_out(requested)
        if note:
            job["note"] = note
        job.update(state="running", stage="抽取点云与轨迹", progress=5,
                   out=str(out), params=dict(params))

        pipeline_cmd = [
            PYTHON, TOOLS / "bag_mapping_pipeline.py",
            "--bag", bag, "--out", out,
            "--voxel", params["voxel"],
            "--min-frames", params["min_frames"],
        ]
        if params["remove_ground"]:
            pipeline_cmd += ["--remove-ground",
                             "--ground-remove-height-m", params["height_low"]]
        if params["ground_offset"].strip():
            pipeline_cmd += ["--ground-offset", params["ground_offset"].strip()]
        _run(pipeline_cmd, job)

        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        offset = metrics.get("ground_offset_m")
        if offset is None:
            raise RuntimeError("地面偏移未能自动估计，请在参数里手工填写")
        job.update(stage="轨迹体检", progress=45)

        traj = out / "traj_camera_init.tum"
        steps = _max_step(traj)
        job["health"] = steps
        if steps["problems"] and not params["force"]:
            raise RuntimeError("轨迹体检未通过：" + "；".join(steps["problems"]))

        job.update(stage="投影 2D 地图", progress=65)
        projector_cmd = [
            PYTHON, TOOLS / "slope_map_projector.py",
            "--pcd", out / "cloud_static.pcd",
            "--trajectory", traj,
            "--ground-offset-m", f"{float(offset):.4f}",
            "--height-low-m", params["height_low"],
            "--height-high-m", params["height_high"],
            "--resolution", params["resolution"],
            "--max-route-distance-m", params["max_route_distance"],
            "--overwrite",
            "--out-prefix", out / "map",
        ]
        if not params["raycast"]:
            projector_cmd.append("--no-raycast-free")
        _run(projector_cmd, job)

        job.update(state="done", stage="完成", progress=100,
                   ground_offset=offset,
                   map_png=_preview(out / "map.pgm"),
                   outfolder=str(out))
    except Exception as exc:                      # noqa: BLE001
        job.update(state="failed", stage="失败", error=str(exc))
        job["log"].append(f"❌ {exc}")


def _max_step(traj_path):
    import numpy as np
    traj = np.loadtxt(traj_path)
    if traj.ndim == 1:
        traj = traj[None, :]
    steps = np.hypot(np.diff(traj[:, 1]), np.diff(traj[:, 2]))
    step_max = float(steps.max()) if len(steps) else 0.0
    z_span = float(np.ptp(traj[:, 3]))
    problems = []
    if step_max > 1.0:
        problems.append(f"相邻位姿最大位移 {step_max:.2f} m（里程计疑似发散）")
    if z_span > 5.0:
        problems.append(f"全程高差 {z_span:.2f} m（里程计疑似发散）")
    return {"poses": int(len(traj)), "max_step_m": round(step_max, 4),
            "z_span_m": round(z_span, 4), "problems": problems}


def _preview(pgm_path):
    """PGM → PNG，供浏览器直接显示。"""
    if not pgm_path.is_file():
        return None
    png = pgm_path.with_suffix(".png")
    if not png.is_file() or png.stat().st_mtime < pgm_path.stat().st_mtime:
        subprocess.run([PYTHON, "-c",
                        "from PIL import Image;Image.open(r'%s').convert('L').save(r'%s')"
                        % (pgm_path, png)], check=False)
    return str(png)


def _can_write(path):
    """能否在该目录创建文件（沙箱里的进程只能写工作区）。"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _resolve_out(requested):
    """目标目录写不了就落到工作区 out/ 下，返回 (实际目录, 提示语)。"""
    if _can_write(requested):
        return requested, None
    fallback = REPO / "out" / (requested.name or "result")
    suffix = 1
    while fallback.exists() and any(fallback.iterdir()):
        suffix += 1
        fallback = REPO / "out" / f"{requested.name or 'result'}_{suffix}"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback, (f"原目录 {requested} 不可写（服务运行在受限环境，只能写工作区），"
                      f"本次结果已存到 {fallback}；要直接写桌面，请双击 tools/启动点云工作台.command 重启服务。")


# ------------------------------------------------------------------ HTTP 层

class Handler(BaseHTTPRequestHandler):
    server_version = "BagStudio/1.0"

    def log_message(self, *_args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):                                        # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            job_id = (parse_qs(parsed.query).get("id") or [""])[0]
            job = JOBS.get(job_id)
            if not job:
                self._send(404, {"error": "no such job"})
                return
            self._send(200, {
                "state": job["state"], "stage": job.get("stage", ""),
                "progress": job.get("progress", 0),
                "log": job["log"][-400:],
                "error": job.get("error"),
                "health": job.get("health"),
                "ground_offset": job.get("ground_offset"),
                "map_png": job.get("map_png"),
                "out": job.get("out"),
                "params": job.get("params", {}),
                "note": job.get("note"),
                "files": job.get("files", []),
            })
            return
        if parsed.path == "/file":
            self._serve_file((parse_qs(parsed.query).get("path") or [""])[0])
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):                                       # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            payload = {}

        if parsed.path == "/api/pick":
            self._send(200, {"path": _pick_path(payload.get("kind", "bag"))})
            return
        if parsed.path == "/api/run":
            job_id = uuid.uuid4().hex[:12]
            with JOB_LOCK:
                JOBS[job_id] = {"state": "queued", "stage": "排队", "progress": 0, "log": []}
            threading.Thread(target=run_job, args=(job_id, payload), daemon=True).start()
            self._send(200, {"id": job_id})
            return
        if parsed.path == "/api/reveal":
            path = Path(payload.get("path", "")).expanduser()
            if path.exists():
                subprocess.run(["open", "-R" if path.is_file() else "", str(path)], check=False)
            self._send(200, {"ok": True})
            return
        self._send(404, {"error": "not found"})

    def _serve_file(self, raw_path):
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in SERVED_SUFFIXES:
            self._send(404, {"error": "not served"})
            return
        ctype = {".png": "image/png", ".pgm": "image/x-portable-graymap"}.get(
            path.suffix.lower(), "application/octet-stream")
        self._send(200, path.read_bytes(), ctype)


def _pick_path(kind):
    if kind == "folder":
        script = 'POSIX path of (choose folder with prompt "选择输出目录")'
    else:
        script = ('POSIX path of (choose file with prompt "选择 rosbag" '
                  'of type {"bag"})')
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True,
                             text=True, timeout=300)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:                                        # noqa: BLE001
        return ""


# ------------------------------------------------------------------ 页面

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>点云工作台</title>
<style>
  :root { --line:#dcdfe3; --ink:#1c1f23; --sub:#61676e; --accent:#0b6bcb; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.55 -apple-system,"PingFang SC",Helvetica,Arial,sans-serif;
         color:var(--ink); background:#fff; }
  header { padding:14px 22px; border-bottom:1px solid var(--line); }
  header h1 { margin:0; font-size:16px; font-weight:600; }
  header p { margin:3px 0 0; color:var(--sub); font-size:12px; }
  main { display:grid; grid-template-columns:380px 1fr; gap:0; min-height:calc(100vh - 62px); }
  .panel { padding:18px 22px; border-right:1px solid var(--line); }
  .panel:last-child { border-right:none; }
  label { display:block; font-size:12px; color:var(--sub); margin:12px 0 4px; }
  input[type=text], input[type=number], select {
    width:100%; padding:7px 9px; border:1px solid var(--line); border-radius:6px;
    font:inherit; color:var(--ink); background:#fff; }
  .row { display:flex; gap:8px; align-items:center; }
  button { font:inherit; padding:7px 14px; border-radius:6px; border:1px solid var(--line);
           background:#fff; cursor:pointer; }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  button:disabled { opacity:.5; cursor:default; }
  .muted { color:var(--sub); font-size:12px; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .chk { display:flex; align-items:center; gap:7px; margin-top:9px; font-size:13px; color:var(--ink); }
  pre#log { margin:0; height:210px; overflow:auto; background:#f6f7f9; border:1px solid var(--line);
            border-radius:8px; padding:10px; font:12px/1.5 ui-monospace,Menlo,monospace; white-space:pre-wrap; }
  .bar { height:6px; background:#eef0f3; border-radius:4px; overflow:hidden; margin:10px 0 6px; }
  .bar > i { display:block; height:100%; width:0; background:var(--accent); transition:width .3s; }
  .stats { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:12px 0; }
  .stat { border:1px solid var(--line); border-radius:8px; padding:9px 11px; }
  .stat b { display:block; font-size:17px; font-weight:600; }
  .stat span { color:var(--sub); font-size:11px; }
  img.map { max-width:100%; border:1px solid var(--line); border-radius:8px; }
  .files a { display:inline-block; margin:0 10px 8px 0; font-size:13px; color:var(--accent); }
  h2 { font-size:13px; font-weight:600; margin:18px 0 6px; }
  .err { color:#b3261e; font-size:13px; margin-top:8px; }
</style></head>
<body>
<header>
  <h1>点云工作台</h1>
  <p>选一个 rosbag → 出 3D 点云 + 2D 地图。本机运行，不联网。</p>
</header>
<main>
  <section class="panel">
    <label>rosbag 路径</label>
    <div class="row">
      <input id="bag" type="text" placeholder="/path/to/scan.bag">
      <button onclick="pick('bag')">选择…</button>
    </div>

    <label>输出目录</label>
    <div class="row">
      <input id="out" type="text" value="out/跑次名">
      <button onclick="pick('folder')">选择…</button>
    </div>
    <div class="muted" style="margin-top:4px">建议放在工作区 out/ 下；写桌面等外部目录需自己双击 .command 启动服务</div>

    <h2>参数</h2>
    <div class="grid2">
      <div><label>帧数阈值</label><input id="min_frames" type="number" value="3" min="2" max="20"></div>
      <div><label>体素 (m)</label><input id="voxel" type="number" value="0.05" step="0.01"></div>
    </div>
    <div class="grid2">
      <div><label>障碍窗下沿 (m)</label><input id="height_low" type="number" value="0.10" step="0.05"></div>
      <div><label>障碍窗上沿 (m)</label><input id="height_high" type="number" value="1.00" step="0.05"></div>
    </div>
    <div class="grid2">
      <div><label>地图分辨率 (m)</label><input id="resolution" type="number" value="0.05" step="0.01"></div>
      <div><label>轨迹距离门限 (m)</label><input id="max_route_distance" type="number" value="8" step="1"></div>
    </div>
    <label>地面偏移（留空=自动估计）</label>
    <input id="ground_offset" type="text" placeholder="自动（实测约 1.174）">

    <div class="chk"><input id="remove_ground" type="checkbox" checked><span>点云去地面</span></div>
    <div class="chk"><input id="raycast" type="checkbox" checked><span>2D 图填充可通行区</span></div>
    <div class="chk"><input id="force" type="checkbox"><span>跳过轨迹体检（发散数据也出图）</span></div>

    <div style="margin-top:18px"><button id="run" class="primary" onclick="start()">开始处理</button></div>
    <div class="err" id="err"></div>
    <div class="muted" id="note" style="margin-top:8px"></div>
  </section>

  <section class="panel">
    <div class="bar"><i id="prog"></i></div>
    <div class="muted" id="stage">等待开始</div>

    <div class="stats" id="stats" style="display:none">
      <div class="stat"><b id="s_poses">–</b><span>位姿数</span></div>
      <div class="stat"><b id="s_step">–</b><span>最大帧间位移</span></div>
      <div class="stat"><b id="s_off">–</b><span>地面偏移 (m)</span></div>
    </div>

    <div id="result" style="display:none">
      <h2>2D 地图</h2>
      <div class="muted" id="usedparams" style="margin-bottom:8px"></div>
      <img class="map" id="mapimg" src="">
      <h2>3D 点云与地图文件</h2>
      <div class="files" id="files"></div>
      <button onclick="reveal()">在 Finder 中显示输出目录</button>
    </div>

    <h2>运行日志</h2>
    <pre id="log"></pre>
  </section>
</main>

<script>
let jobId = null, timer = null, outDir = "";

function val(id) { return document.getElementById(id).value.trim(); }
function num(id) { return parseFloat(val(id)); }
function chk(id) { return document.getElementById(id).checked; }

async function pick(kind) {
  const r = await fetch('/api/pick', {method:'POST', headers:{'Content-Type':'application/json'},
                                      body: JSON.stringify({kind})});
  const d = await r.json();
  if (!d.path) return;
  if (kind === 'bag') {
    document.getElementById('bag').value = d.path;
    if (val('out') === 'out/跑次名' || !val('out')) {
      const base = d.path.split('/').pop().replace(/\.bag$/, '');
      document.getElementById('out').value = 'out/' + base;
    }
  } else {
    document.getElementById('out').value = d.path;
  }
}

async function start() {
  document.getElementById('err').textContent = '';
  document.getElementById('result').style.display = 'none';
  document.getElementById('stats').style.display = 'none';
  document.getElementById('run').disabled = true;
  const payload = {
    bag: val('bag'), out: val('out'),
    min_frames: String(Math.round(num('min_frames'))),
    voxel: String(num('voxel')),
    height_low: String(num('height_low')),
    height_high: String(num('height_high')),
    resolution: String(num('resolution')),
    max_route_distance: String(num('max_route_distance')),
    ground_offset: val('ground_offset'),
    remove_ground: chk('remove_ground'),
    raycast: chk('raycast'),
    force: chk('force'),
  };
  const r = await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'},
                                     body: JSON.stringify(payload)});
  const d = await r.json();
  jobId = d.id;
  timer = setInterval(poll, 700);
}

async function poll() {
  if (!jobId) return;
  const r = await fetch('/api/status?id=' + jobId);
  const d = await r.json();
  document.getElementById('prog').style.width = (d.progress || 0) + '%';
  document.getElementById('stage').textContent = d.state === 'running' ? d.stage :
      (d.state === 'done' ? '完成' : (d.state === 'failed' ? '失败' : d.stage));
  document.getElementById('log').textContent = (d.log || []).join('\n');
  document.getElementById('note').textContent = d.note || '';
  document.getElementById('log').scrollTop = 1e9;

  if (d.health) {
    document.getElementById('stats').style.display = 'grid';
    document.getElementById('s_poses').textContent = d.health.poses;
    document.getElementById('s_step').textContent = d.health.max_step_m + ' m';
    document.getElementById('s_off').textContent = (d.ground_offset ?? '–');
  }
  if (d.state === 'done') {
    clearInterval(timer);
    document.getElementById('run').disabled = false;
    outDir = d.out;
    const img = document.getElementById('mapimg');
    img.src = '/file?path=' + encodeURIComponent(d.map_png) + '&t=' + Date.now();
    const p = d.params || {};
    document.getElementById('usedparams').textContent =
      '本次参数：帧数≥' + p.min_frames + '｜体素 ' + p.voxel + ' m｜障碍窗 ' +
      p.height_low + '~' + p.height_high + ' m｜分辨率 ' + p.resolution +
      ' m｜门限 ' + p.max_route_distance + ' m｜去地面 ' + (p.remove_ground ? '是' : '否') +
      '｜射线填充 ' + (p.raycast ? '是' : '否');
    document.getElementById('result').style.display = 'block';
    const files = [
      ['cloud_static.pcd', '3D 点云（静态）'],
      ['cloud_noground.pcd', '3D 点云（去地面）'],
      ['cloud_voxel_all.pcd', '3D 点云（全量）'],
      ['map.pgm', '2D 地图 PGM'],
      ['map.yaml', '2D 地图 YAML'],
    ];
    document.getElementById('files').innerHTML = files.map(
      ([f, label]) => '<a href="/file?path=' + encodeURIComponent(d.out + '/' + f) +
                      '" target="_blank">' + label + '</a>').join('');
  }
  if (d.state === 'failed') {
    clearInterval(timer);
    document.getElementById('run').disabled = false;
    document.getElementById('err').textContent = d.error || '失败';
  }
}

async function reveal() {
  await fetch('/api/reveal', {method:'POST', headers:{'Content-Type':'application/json'},
                              body: JSON.stringify({path: outDir})});
}
</script>
</body></html>
"""


def main():
    parser = argparse.ArgumentParser(description="点云工作台（本机 Web UI）")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"点云工作台已启动：{url}")
    print("按 Ctrl-C 停止")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
