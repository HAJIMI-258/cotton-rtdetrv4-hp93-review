
import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EPOCH_RE = re.compile(r"Epoch:\s*\[(\d+)\]\s*\[(\d+)\s*/\s*(\d+)\].*?eta:\s*([^\s]+).*?lr:\s*([0-9.eE+-]+).*?loss:\s*([0-9.]+)\s*\(([0-9.]+)\)")

HTML_PAGE = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Cotton Baseline Monitor</title>
<style>
:root{--bg:#08111f;--panel:#111d2f;--panel2:#16263d;--line:#284360;--text:#eef7ff;--muted:#a6bdd5;--green:#32e875;--cyan:#34d5ff;--blue:#629bff;--pink:#ff5fa2;--orange:#ffb238;--red:#ff4d5e;--purple:#b27cff;}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0%,#143b5e 0,#08111f 34%,#070c16 100%);color:var(--text);font-family:Segoe UI,Microsoft YaHei,Arial,sans-serif;letter-spacing:0}.wrap{padding:26px 30px 36px}.top{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;margin-bottom:20px}.title h1{font-size:34px;margin:0 0 6px;font-weight:800}.title p{margin:0;color:var(--muted);font-size:15px}.pill{display:inline-flex;align-items:center;gap:8px;padding:9px 14px;border:1px solid var(--line);background:rgba(17,29,47,.82);border-radius:999px;color:var(--muted);font-weight:600}.dot{width:10px;height:10px;border-radius:50%;background:var(--green);box-shadow:0 0 18px var(--green)}.dot.stop{background:var(--red);box-shadow:0 0 18px var(--red)}.cards{display:grid;grid-template-columns:repeat(6,minmax(150px,1fr));gap:14px}.card{background:linear-gradient(180deg,rgba(22,38,61,.96),rgba(12,24,41,.94));border:1px solid var(--line);border-radius:12px;padding:16px 18px;min-height:116px;box-shadow:0 12px 30px rgba(0,0,0,.22)}.label{font-size:13px;color:var(--muted);margin-bottom:8px}.value{font-size:26px;font-weight:800;line-height:1.15}.sub{margin-top:6px;color:#d5e7ff;font-size:14px;line-height:1.35}.green{color:var(--green)}.cyan{color:var(--cyan)}.orange{color:var(--orange)}.pink{color:var(--pink)}.blue{color:var(--blue)}.purple{color:var(--purple)}.bar{height:11px;background:#22334b;border-radius:999px;overflow:hidden;margin-top:12px}.fill{height:100%;background:linear-gradient(90deg,var(--green),var(--cyan),var(--blue));width:0%;transition:width .3s}.section{margin-top:24px}.section h2{font-size:22px;margin:0 0 12px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}.tablebox,.logbox{background:rgba(10,19,34,.92);border:1px solid var(--line);border-radius:12px;overflow:hidden}.tablebox{padding:0}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:11px 13px;border-bottom:1px solid rgba(63,94,128,.55);white-space:nowrap}th{color:#abd0f4;background:rgba(30,52,83,.7);font-weight:800}tr:last-child td{border-bottom:0}.logbox pre{margin:0;padding:16px;max-height:430px;overflow:auto;font:13px/1.45 Consolas,monospace;color:#eaf4ff;white-space:pre-wrap}.warn{color:var(--orange)}@media(max-width:1300px){.cards{grid-template-columns:repeat(3,1fr)}.grid2{grid-template-columns:1fr}}@media(max-width:760px){.wrap{padding:18px}.cards{grid-template-columns:1fr}.top{display:block}.pill{margin-top:12px}.title h1{font-size:26px}}
</style>
</head>
<body><div class="wrap">
<div class="top"><div class="title"><h1>Cotton RT-DETRv4 基线监控</h1><p>只读同步训练日志、验证指标与 GPU 状态；每 2 秒刷新一次</p></div><div class="pill"><span id="dot" class="dot"></span><span id="clock">--</span></div></div>
<div class="cards">
  <div class="card"><div class="label">状态</div><div id="status" class="value green">--</div><div id="delay" class="sub">--</div></div>
  <div class="card"><div class="label">当前进度</div><div id="progress" class="value cyan">--</div><div id="iter" class="sub">--</div><div class="bar"><div id="bar" class="fill"></div></div></div>
  <div class="card"><div class="label">当前 loss / lr</div><div id="loss" class="value orange">--</div><div id="lr" class="sub">--</div></div>
  <div class="card"><div class="label">最新验证</div><div id="latest" class="value blue">--</div><div id="latest_sub" class="sub">--</div></div>
  <div class="card"><div class="label">最佳 AP50</div><div id="best" class="value pink">--</div><div id="best_sub" class="sub">--</div></div>
  <div class="card"><div class="label">GPU / 温度</div><div id="gpu" class="value purple">--</div><div id="gpu_sub" class="sub">--</div></div>
</div>
<div class="section"><h2>验证指标</h2><div class="tablebox"><table><thead><tr><th>epoch</th><th>mAP</th><th>AP50</th><th>AP75</th><th>APS</th><th>APM</th><th>APL</th></tr></thead><tbody id="metrics"><tr><td colspan="7">读取中...</td></tr></tbody></table></div></div>
<div class="section grid2"><div><h2>训练日志尾部</h2><div class="logbox"><pre id="stdout">读取中...</pre></div></div><div><h2>JSON 日志尾部</h2><div class="logbox"><pre id="jsonlog">读取中...</pre></div></div></div>
</div>
<script>
function f4(x){return (x===null||x===undefined||isNaN(x))?'--':Number(x).toFixed(4)}
async function tick(){
  try{
    const r=await fetch('/api/status?ts='+Date.now());
    const d=await r.json();
    document.getElementById('clock').textContent='更新时间 '+d.now;
    document.getElementById('status').textContent=d.running?'RUNNING':'STOPPED';
    document.getElementById('status').className='value '+(d.running?'green':'warn');
    document.getElementById('dot').className='dot '+(d.running?'':'stop');
    document.getElementById('delay').textContent='日志延迟 '+d.log_age_sec+' 秒';
    const cur=d.current||{};
    document.getElementById('progress').textContent='Epoch '+(cur.epoch??'--')+' / '+d.total_epochs;
    document.getElementById('iter').textContent=(cur.iter??'--')+' / '+(cur.total_iter??'--')+' · ETA '+(cur.eta??'--');
    document.getElementById('bar').style.width=(cur.percent||0)+'%';
    document.getElementById('loss').textContent=cur.loss?Number(cur.loss).toFixed(4):'--';
    document.getElementById('lr').textContent='avg '+(cur.loss_avg?Number(cur.loss_avg).toFixed(4):'--')+' · lr '+(cur.lr??'--');
    const latest=d.latest_metric||{}; const best=d.best_ap50||{};
    document.getElementById('latest').textContent=f4(latest.ap50);
    document.getElementById('latest_sub').textContent='epoch '+(latest.epoch??'--')+' · mAP '+f4(latest.map)+' · AP75 '+f4(latest.ap75);
    document.getElementById('best').textContent=f4(best.ap50);
    document.getElementById('best_sub').textContent='epoch '+(best.epoch??'--')+' · mAP '+f4(best.map)+' · AP75 '+f4(best.ap75);
    const g=d.gpu||{};
    document.getElementById('gpu').textContent=(g.util??'--')+'% / '+(g.temp??'--')+'C';
    document.getElementById('gpu_sub').textContent=(g.mem_used??'--')+'/'+(g.mem_total??'--')+' MiB · '+(g.power??'--')+' W';
    let rows=(d.metrics||[]).slice().reverse().map(m=>`<tr><td>${m.epoch}</td><td>${f4(m.map)}</td><td>${f4(m.ap50)}</td><td>${f4(m.ap75)}</td><td>${f4(m.aps)}</td><td>${f4(m.apm)}</td><td>${f4(m.apl)}</td></tr>`).join('');
    document.getElementById('metrics').innerHTML=rows||'<tr><td colspan="7">暂无验证结果</td></tr>';
    document.getElementById('stdout').textContent=d.stdout_tail||'';
    document.getElementById('jsonlog').textContent=d.json_tail||'';
  }catch(e){document.getElementById('status').textContent='ERROR';document.getElementById('status').className='value warn';document.getElementById('stdout').textContent=String(e);}
}
tick();setInterval(tick,2000);
</script></body></html>'''

def tail_text(path, max_bytes=90000):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
        return data.decode('utf-8', 'replace')
    except Exception as e:
        return f"NO_READ {path}: {e}"

def parse_metrics(log_text):
    out=[]
    for line in log_text.splitlines():
        line=line.strip()
        if not line.startswith('{'):
            continue
        try:
            obj=json.loads(line)
        except Exception:
            continue
        arr=obj.get('test_coco_eval_bbox')
        if isinstance(arr, list) and len(arr)>=6:
            out.append({'epoch': obj.get('epoch'), 'map': arr[0], 'ap50': arr[1], 'ap75': arr[2], 'aps': arr[3], 'apm': arr[4], 'apl': arr[5]})
    return out

def parse_current(stdout_text):
    matches=list(EPOCH_RE.finditer(stdout_text))
    if not matches:
        return None
    m=matches[-1]
    epoch=int(m.group(1)); it=int(m.group(2)); total=int(m.group(3))
    return {'epoch': epoch, 'iter': it, 'total_iter': total, 'eta': m.group(4), 'lr': m.group(5), 'loss': float(m.group(6)), 'loss_avg': float(m.group(7)), 'percent': round(100*it/max(total,1), 2)}

def gpu_info():
    try:
        cmd=['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader,nounits']
        p=subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        line=(p.stdout or '').strip().splitlines()[0]
        parts=[x.strip() for x in line.split(',')]
        return {'util': parts[0], 'mem_used': parts[1], 'mem_total': parts[2], 'temp': parts[3], 'power': parts[4]}
    except Exception as e:
        return {'error': str(e)}

def is_running():
    try:
        p=subprocess.run(['powershell','-NoProfile','-Command',"Get-CimInstance Win32_Process | ? {$_.CommandLine -like '*balanced_clean_baseline*' -and $_.CommandLine -like '*train.py*'} | Select -First 1 -ExpandProperty ProcessId"], capture_output=True, text=True, timeout=8)
        return bool((p.stdout or '').strip())
    except Exception:
        return False

class Handler(BaseHTTPRequestHandler):
    output_dir = ''
    total_epochs = 120
    def log_message(self, fmt, *args):
        return
    def do_GET(self):
        if self.path.startswith('/api/status'):
            self.send_json(self.status())
        else:
            data=HTML_PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data)
    def send_json(self, obj):
        data=json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def status(self):
        od=self.output_dir
        log_path=os.path.join(od,'log.txt')
        stdout_path=os.path.join(od,'train_stdout.log')
        log_tail=tail_text(log_path, 120000)
        stdout_tail=tail_text(stdout_path, 90000)
        metrics=parse_metrics(log_tail)
        latest=metrics[-1] if metrics else None
        best=max(metrics, key=lambda x: (x.get('ap50') if x.get('ap50') is not None else -1), default=None)
        age='--'
        try:
            age=int(time.time()-max(os.path.getmtime(log_path), os.path.getmtime(stdout_path)))
        except Exception:
            pass
        return {'now': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'running': is_running(), 'total_epochs': self.total_epochs, 'log_age_sec': age, 'current': parse_current(stdout_tail), 'metrics': metrics[-30:], 'latest_metric': latest, 'best_ap50': best, 'gpu': gpu_info(), 'stdout_tail': stdout_tail[-18000:], 'json_tail': log_tail[-12000:]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=18080)
    ap.add_argument('--output', required=True)
    ap.add_argument('--epochs', type=int, default=120)
    args=ap.parse_args()
    Handler.output_dir=args.output
    Handler.total_epochs=args.epochs
    httpd=ThreadingHTTPServer((args.host,args.port), Handler)
    print(f"monitor listening on http://{args.host}:{args.port}, output={args.output}", flush=True)
    httpd.serve_forever()

if __name__ == '__main__':
    main()
