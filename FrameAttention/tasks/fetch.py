"""Small download helpers (standard library only): resume, retries, progress, size checks."""

import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

UA = {"User-Agent": "frame-attention-data-downloader/1.0"}


def human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def remote_size(url):
    """Content-Length of url, or None if the server does not say."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n is not None else None
    except Exception:
        return None


def download(url, path, retries=8, quiet=False):
    """Download url to path. Resumes a partial file (path + '.part'); skips if path exists."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    part = path + ".part"
    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        headers = dict(UA, **({"Range": f"bytes={have}-"} if have else {}))
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                if have and r.status != 206:  # server ignored the range: start over
                    have = 0
                total = r.headers.get("Content-Length")
                total = int(total) + have if total is not None else None
                mode = "ab" if have else "wb"
                done, t0, last = have, time.time(), 0.0
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if not quiet and time.time() - last > 2:
                            last = time.time()
                            rate = (done - have) / max(1e-6, last - t0)
                            pct = f"{100 * done / total:5.1f}%" if total else ""
                            sys.stdout.write(f"\r    {os.path.basename(path)}  {human(done)} {pct}  {human(rate)}/s   ")
                            sys.stdout.flush()
                if total is not None and os.path.getsize(part) != total:
                    raise IOError(f"incomplete: {os.path.getsize(part)} of {total} bytes")
            os.replace(part, path)
            if not quiet:
                sys.stdout.write(f"\r    {os.path.basename(path)}  {human(os.path.getsize(path))}  done{' ' * 30}\n")
            return path
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise
            err = e
        except Exception as e:  # network hiccup: wait and resume
            err = e
        wait = min(60, 2 ** attempt)
        print(f"\n    retry {attempt}/{retries} for {os.path.basename(path)} in {wait}s ({err})")
        time.sleep(wait)
    raise IOError(f"failed to download {url}")


def fetch_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def free_space(path):
    os.makedirs(path, exist_ok=True)
    return shutil.disk_usage(path).free
