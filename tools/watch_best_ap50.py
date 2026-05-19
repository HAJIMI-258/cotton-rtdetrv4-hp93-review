
import json, os, shutil, time, traceback
OUT = r'C:\Users\WYZ\Desktop\cotton\rtdetrv4_hp93\RT-DETRv4\outputs\rtv4_hgnetv2_m_cotton_balanced_v6_768'
LOG = os.path.join(OUT, 'log.txt')
BEST = os.path.join(OUT, 'best_ap50.pth')
BEST_JSON = os.path.join(OUT, 'best_ap50.json')
WATCH_LOG = os.path.join(OUT, 'best_ap50_watcher.log')
POLL = 60

def log(s):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + str(s)
    with open(WATCH_LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)

def read_rows():
    rows = []
    if not os.path.exists(LOG):
        return rows
    with open(LOG, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line=line.strip()
            if not line.startswith('{'):
                continue
            try:
                r=json.loads(line)
            except Exception:
                continue
            m=r.get('test_coco_eval_bbox')
            if isinstance(m, list) and len(m) > 1 and 'epoch' in r:
                rows.append(r)
    return rows

def save_best(row):
    epoch = int(row['epoch'])
    m = row['test_coco_eval_bbox']
    ckpt = os.path.join(OUT, f'checkpoint{epoch:04d}.pth')
    if not os.path.exists(ckpt):
        return False, f'missing checkpoint {ckpt}'
    meta = {
        'epoch': epoch,
        'mAP': m[0],
        'AP50': m[1],
        'AP75': m[2],
        'checkpoint': os.path.basename(ckpt),
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    old = None
    if os.path.exists(BEST_JSON):
        try:
            with open(BEST_JSON, 'r', encoding='utf-8') as f:
                old = json.load(f)
        except Exception:
            old = None
    if old and old.get('epoch') == epoch and abs(float(old.get('AP50', -1)) - float(m[1])) < 1e-12 and os.path.exists(BEST):
        return False, 'unchanged'
    tmp = BEST + '.tmp'
    shutil.copy2(ckpt, tmp)
    os.replace(tmp, BEST)
    named = os.path.join(OUT, f'best_ap50_epoch{epoch:04d}_ap50_{m[1]:.6f}.pth')
    if not os.path.exists(named):
        shutil.copy2(ckpt, named)
    with open(BEST_JSON + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(BEST_JSON + '.tmp', BEST_JSON)
    return True, meta

def once():
    rows = read_rows()
    if not rows:
        return False, 'no rows'
    best = max(rows, key=lambda r: r['test_coco_eval_bbox'][1])
    return save_best(best)

if __name__ == '__main__':
    log('watcher started')
    while True:
        try:
            changed, info = once()
            if changed:
                log('UPDATED ' + json.dumps(info, ensure_ascii=False))
            elif info != 'unchanged':
                log('CHECK ' + str(info))
        except Exception as e:
            log('ERROR ' + repr(e))
            log(traceback.format_exc())
        time.sleep(POLL)
