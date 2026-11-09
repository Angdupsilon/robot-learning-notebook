"""
Download or generate every dataset into ./data with one command.

    python download_data.py                                    # everything
    python download_data.py --only shapes boxes                # generated data only (no internet needed)
    python download_data.py --only clevrtex --clevrtex-parts 1 # smoke test: 1 of the 5 CLEVRTex parts
    python download_data.py --only movi_a --movi-shards 4      # smoke test: 4 MOVi-A training shards

  dataset          how         used for                          ends up in
  shapes           generated   Stage 1 binding test              data/shapes/
  boxes            generated   binding in text (entity tracking) data/boxes/
  tinyshakespeare  download    frame_gpt.py demo                 data/tinyshakespeare/input.txt
  clevrtex         download    Stage 2: grouping + binding       data/clevrtex/
  movi_a           download    Stage 2: common fate (variant E)  data/movi_a/

Re-running skips whatever is already finished. MOVi conversion needs TensorFlow; if it is not
installed, a private environment is created in data/.tfds-venv (training never needs it).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from tasks import boxes, clevrtex, movi, shapes  # noqa: E402
from tasks.fetch import download, free_space, human, remote_size  # noqa: E402

DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
ALL = ["shapes", "boxes", "tinyshakespeare", "clevrtex", "movi_a"]


def finished(path, tag):
    p = os.path.join(path, ".done")
    return os.path.exists(p) and open(p).read() == json.dumps(tag)


def mark(path, tag):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, ".done"), "w") as f:
        f.write(json.dumps(tag))


def du(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


def check_space(need, where, force):
    have = free_space(where)
    print(f"    needs about {human(need)}; free on disk: {human(have)}")
    if need > have and not force:
        raise SystemExit("    not enough disk space (use --force to try anyway)")


# ----------------------------------------------------------------------------- generated

def get_shapes(args):
    out = os.path.join(DATA, "shapes")
    if finished(out, "v1"):
        return print("  shapes: already done")
    t = time.time()
    shapes.generate(out, workers=args.workers)
    mark(out, "v1")
    print(f"  shapes: done in {time.time() - t:.0f}s ({human(du(out))})")


def get_boxes(args):
    out = os.path.join(DATA, "boxes")
    if finished(out, "v1"):
        return print("  boxes: already done")
    t = time.time()
    boxes.generate(out)
    mark(out, "v1")
    print(f"  boxes: done in {time.time() - t:.0f}s ({human(du(out))})")


def get_tinyshakespeare(args):
    path = os.path.join(DATA, "tinyshakespeare", "input.txt")
    local = os.path.join(ROOT, "..", "nanoGPT", "ShakespeareText.txt")  # already in robot-learning-notebook
    if not os.path.exists(path) and os.path.exists(local):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copyfile(local, path)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    download(url, path)  # no-op if the file is already there
    print(f"  tinyshakespeare: {path} ({human(os.path.getsize(path))})")


# ----------------------------------------------------------------------------- CLEVRTex

def extract(archive, dest):
    if shutil.which("tar"):
        subprocess.run(["tar", "-xzf", archive, "-C", dest], check=True)
    else:
        with tarfile.open(archive) as t:
            kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            t.extractall(dest, **kw)


def get_clevrtex(args):
    out, raw = os.path.join(DATA, "clevrtex"), os.path.join(RAW, "clevrtex")
    names = clevrtex.ARCHIVES[:args.clevrtex_parts] + clevrtex.ARCHIVES[5:]
    tag = {"archives": names, "res": clevrtex.DEFAULTS["res"]}
    if finished(out, tag):
        return print("  clevrtex: already done")
    todo = [n for n in names if not os.path.exists(os.path.join(raw, ".extracted", n))]
    if todo:
        sizes = [remote_size(f"{clevrtex.BASE_URL}/{n}") for n in todo]
        known = sum(s for s in sizes if s)
        print(f"    {len(todo)} archives to fetch, {human(known)} compressed")
        check_space(int(2.3 * known) + (5 << 30), DATA, args.force)  # archive + extracted + converted
    for n in names:
        flag = os.path.join(raw, ".extracted", n)
        if os.path.exists(flag):
            continue
        archive = download(f"{clevrtex.BASE_URL}/{n}", os.path.join(raw, "archives", n))
        print(f"    extracting {n}")
        extract(archive, raw)
        os.makedirs(os.path.dirname(flag), exist_ok=True)
        open(flag, "w").close()
        if not args.keep_archives:
            os.remove(archive)
    print("    converting (crop, resize, questions)")
    clevrtex.convert(raw, out, workers=args.workers)
    mark(out, tag)
    print(f"  clevrtex: done ({human(du(out))}); raw files in {raw} ({human(du(raw))})")


# ----------------------------------------------------------------------------- MOVi

def tfds_python():
    """A python that can import tensorflow + tensorflow_datasets: this one, or a private venv."""
    try:
        import tensorflow  # noqa: F401
        import tensorflow_datasets  # noqa: F401
        return sys.executable, dict(os.environ)
    except ImportError:
        pass
    venv = os.path.join(DATA, ".tfds-venv")
    py = os.path.join(venv, "bin", "python")
    pkgs = ["tensorflow", "tensorflow-datasets", "numpy"]
    if not os.path.exists(py):
        print("    TensorFlow not found: creating data/.tfds-venv (one time, a few minutes)")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv], check=True)
        except subprocess.CalledProcessError:
            target = os.path.join(DATA, ".tfds-pkgs")
            print("    venv unavailable; installing into data/.tfds-pkgs instead")
            subprocess.run([sys.executable, "-m", "pip", "install", "--target", target] + pkgs, check=True)
            env = dict(os.environ, PYTHONPATH=target + os.pathsep + os.environ.get("PYTHONPATH", ""))
            return sys.executable, env
    subprocess.run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    subprocess.run([py, "-m", "pip", "install", "-q"] + pkgs, check=True)
    return py, dict(os.environ)


def get_movi(args, name="movi_a"):
    out = os.path.join(DATA, name)
    tag = {"shards": args.movi_shards, "cfg": movi.DEFAULTS}
    if finished(out, tag):
        return print(f"  {name}: already done")
    vdir = os.path.join(RAW, name, "128x128", "1.0.0")
    info_path = os.path.join(vdir, "dataset_info.json")
    if not args.movi_shards:
        download(f"{movi.GCS}/{name}/128x128/1.0.0/dataset_info.json", info_path, quiet=True)
        with open(info_path) as f:
            total = sum(int(s.get("numBytes", 0)) for s in json.load(f).get("splits", []))
        if total:
            check_space(total + (3 << 30), DATA, args.force)
    vdir = movi.download(name, "128x128", RAW, max_shards=args.movi_shards or None)
    py, env = tfds_python()
    d = movi.DEFAULTS
    print("    converting with tensorflow-datasets")
    subprocess.run([py, "-m", "tasks.movi", "convert", "--src", vdir, "--out", out, "--frame", str(d["frame"]),
                    "--flow-steps", str(d["flow_steps"]), "--grid", str(d["grid"])], check=True, cwd=ROOT, env=env)
    print("    making questions")
    movi.prepare(out)
    mark(out, tag)
    print(f"  {name}: done ({human(du(out))}); raw shards in {vdir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=ALL, default=ALL)
    p.add_argument("--clevrtex-parts", type=int, default=5, choices=range(1, 6),
                   help="how many of the 5 CLEVRTex parts to fetch (OOD and CAMO always)")
    p.add_argument("--movi-shards", type=int, default=0, help="only this many MOVi training shards (0 = all)")
    p.add_argument("--workers", type=int, default=None, help="processes for generation/conversion")
    p.add_argument("--keep-archives", action="store_true", help="keep CLEVRTex .tar.gz after extracting")
    p.add_argument("--force", action="store_true", help="continue even if the disk looks too small")
    args = p.parse_args()
    os.makedirs(DATA, exist_ok=True)
    steps = dict(shapes=get_shapes, boxes=get_boxes, tinyshakespeare=get_tinyshakespeare,
                 clevrtex=get_clevrtex, movi_a=get_movi)
    for name in ALL:
        if name in args.only:
            print(f"[{name}]")
            steps[name](args)
    print("\nAll requested datasets are ready in", DATA)


if __name__ == "__main__":
    main()
