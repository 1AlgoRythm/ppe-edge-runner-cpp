"""Python reference for the C++ runner: same models, same preprocessing, same thresholds.

Written independently of the C++ code (numpy + cv2.dnn NMS) so that agreement between the two means something.
Modes:  --dump FILE   detections per image  (compared with the C++ dump by python/compare.py)
        --bench       per-stage timing over the same images, same JSON schema as `ppe_runner --bench`
"""
import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PERSON_NAMES = ["person"]
PPE_NAMES = ["hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit"]


def letterbox(img, size=640, override=None):
    """Resize keeping aspect ratio, pad to size x size with grey 114. Returns (img, scale, pad_x, pad_y).

    override: path of a letterboxed PNG written by the C++ runner. The geometry (scale, padding) is still computed here;
    only the pixels are taken from the file. Used to separate pipeline logic from OpenCV-version resize differences.
    """
    h, w = img.shape[:2]                                   # (H, W, 3) BGR
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    bottom, right = size - nh - top, size - nw - left
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    if override is not None:
        out = cv2.imread(override)
        assert out.shape == (size, size, 3), override
    return out, scale, left, top                           # (size, size, 3)


def to_tensor(lb):
    """BGR HWC uint8 -> RGB NCHW float32 in [0, 1]."""
    chw = np.transpose(lb[:, :, ::-1], (2, 0, 1))          # (3, 640, 640) RGB
    return np.ascontiguousarray(chw[None], dtype=np.float32) / np.float32(255)  # (1, 3, 640, 640)


def decode(output, scale, pad_x, pad_y, orig_w, orig_h, conf, iou):
    """output (1, 4 + nc, 8400) -> list of (x1, y1, x2, y2, score, cls) in original pixels."""
    pred = output[0].astype(np.float32).T                  # (8400, 4 + nc)
    scores_all = pred[:, 4:]                               # (8400, nc)
    cls = scores_all.argmax(axis=1)
    score = scores_all[np.arange(len(cls)), cls]
    keep = score >= conf
    pred, cls, score = pred[keep], cls[keep], score[keep]
    if len(score) == 0:
        return []
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = (cx - w / 2 - pad_x) / scale
    y1 = (cy - h / 2 - pad_y) / scale
    boxes = np.stack([x1, y1, w / scale, h / scale], 1)    # (N, 4) x, y, w, h
    idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), score.tolist(), cls.tolist(), conf, iou)
    dets = []
    for i in np.array(idx).reshape(-1):
        bx, by, bw, bh = boxes[i]
        dets.append((max(0.0, float(bx)), max(0.0, float(by)), min(float(orig_w), float(bx + bw)),
                     min(float(orig_h), float(by + bh)), float(score[i]), int(cls[i])))
    return dets


class Detector:
    def __init__(self, path, n_classes, threads, ort_config=()):
        opts = ort.SessionOptions()
        for kv in ort_config:
            opts.add_session_config_entry(*kv.split("=", 1))
        opts.intra_op_num_threads = threads                # 0 = ORT default
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name
        shape = self.sess.get_outputs()[0].shape
        assert shape[1] == 4 + n_classes, f"{path}: output shape {shape}"

    def detect(self, img, conf, iou, lb_path=None):
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        lb, scale, px, py = letterbox(img, override=lb_path)
        x = to_tensor(lb)                                  # (1, 3, 640, 640)
        t1 = time.perf_counter()
        out = self.sess.run(None, {self.name: x})[0]       # (1, 4 + nc, 8400)
        t2 = time.perf_counter()
        dets = decode(out, scale, px, py, w, h, conf, iou)
        t3 = time.perf_counter()
        return dets, ((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)


class Pipeline:
    def __init__(self, person, ppe, cascade, conf, iou, lb_dir=None):
        self.person, self.ppe, self.cascade, self.conf, self.iou, self.lb_dir = person, ppe, cascade, conf, iou, lb_dir

    def process(self, img, key=""):
        """Returns persons, items, person_t, ppe_t, total_ms, n_crops (stage times are (pre, inf, post))."""
        lb = lambda stage: f"{self.lb_dir}/{key}_{stage}.png" if self.lb_dir else None
        t0 = time.perf_counter()
        zero = (0.0, 0.0, 0.0)
        if not self.cascade:
            d = self.person or self.ppe
            items, tt = d.detect(img, self.conf, self.iou, lb("person" if self.person else "ppe"))
            persons = items if self.person else []
            return persons, items, tt, zero, (time.perf_counter() - t0) * 1e3, 0
        persons, pt = self.person.detect(img, self.conf, self.iou, lb("person"))
        items, qt, n = [], [0.0, 0.0, 0.0], 0
        for x1, y1, x2, y2, _, _ in persons:
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            if ix2 - ix1 < 10 or iy2 - iy1 < 10:
                continue
            crop = img[iy1:iy2, ix1:ix2]                   # (h, w, 3) view
            dets, t = self.ppe.detect(crop, self.conf, self.iou, lb(f"ppe_{n}"))
            items += [(a + ix1, b + iy1, c + ix1, d + iy1, s, k) for a, b, c, d, s, k in dets]
            qt = [qt[i] + t[i] for i in range(3)]
            n += 1
        return persons, items, pt, tuple(qt), (time.perf_counter() - t0) * 1e3, n


def stat(v):
    v = np.asarray(v, dtype=float)
    return {"median": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "mean": float(v.mean()), "min": float(v.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["fp32", "int8"])
    ap.add_argument("--cascade", action="store_true")
    ap.add_argument("--images", default="data/person_test")
    ap.add_argument("--models", default="models")
    ap.add_argument("--dump")
    ap.add_argument("--lb-dir", help="use the C++ runner's letterboxed PNGs as network input (identical-input check)")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ort-config", action="append", default=[], help="extra session config KEY=VALUE (repeatable)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    a = ap.parse_args()

    mdir = Path(a.models)
    person = Detector(mdir / f"person_{a.variant}.onnx", 1, a.threads, a.ort_config)
    ppe = Detector(mdir / f"ppe_{a.variant}.onnx", 7, a.threads, a.ort_config) if a.cascade else None
    pipe = Pipeline(person, ppe, a.cascade, a.conf, a.iou, a.lb_dir)
    mode = "cascade" if a.cascade else "person"
    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    imgs = [cv2.imread(str(p)) for p in files]
    meta = {"lang": "python", "mode": mode, "variant": a.variant,
            "person_model": str(mdir / f"person_{a.variant}.onnx"),
            "ppe_model": str(mdir / f"ppe_{a.variant}.onnx") if a.cascade else "",
            "input": "cpp-letterboxed-png" if a.lb_dir else "own-letterbox",
            "conf": a.conf, "iou": a.iou, "threads": a.threads, "ort_config": ",".join(a.ort_config), "onnxruntime": ort.__version__, "ort_library": str(Path(ort.__file__).parent / "capi" / f"libonnxruntime.{ort.__version__}.dylib"),
            "opencv": cv2.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()}

    if a.dump:
        res = {}
        for p, im in zip(files, imgs):
            persons, items, *_ = pipe.process(im, p.stem)
            res[p.name] = {"pixel_sum": int(im.sum(dtype=np.uint64)),
                           "persons": [list(d) for d in persons], "items": [list(d) for d in items]}
        Path(a.dump).parent.mkdir(parents=True, exist_ok=True)
        Path(a.dump).write_text(json.dumps({"meta": meta, "images": res}))
        print(f"dumped detections ({mode}, {a.variant}) -> {a.dump}")

    if a.bench:
        for _ in range(a.warmup):
            for im in imgs:
                pipe.process(im)
        cols = {k: [] for k in ("pre", "inf", "post", "stages", "e2e", "person", "ppe", "crops", "dets")}
        load0 = os.getloadavg()
        for _ in range(a.passes):
            for im in imgs:
                persons, items, pt, qt, total, n = pipe.process(im)
                t = [pt[i] + qt[i] for i in range(3)]
                for k, v in zip(cols, (t[0], t[1], t[2], sum(t), total, sum(pt), sum(qt), n, len(items))):
                    cols[k].append(v)
        meta["loadavg_before_after"] = [load0[0], os.getloadavg()[0]]
        out = {"meta": meta, "n_images": len(imgs), "warmup_passes": a.warmup, "timed_passes": a.passes,
               "n_timings": len(cols["e2e"]),
               "note": "Per-image times in ms. In cascade mode each stage is summed over the person run and all PPE crops of that image. Image decoding is excluded.",
               "preprocess_ms": stat(cols["pre"]), "inference_ms": stat(cols["inf"]), "postprocess_ms": stat(cols["post"]),
               "stages_total_ms": stat(cols["stages"]), "e2e_ms": stat(cols["e2e"]),
               "person_stage_ms": stat(cols["person"]), "ppe_stage_ms": stat(cols["ppe"]),
               "crops_per_image_mean": float(np.mean(cols["crops"])), "dets_per_image_mean": float(np.mean(cols["dets"]))}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"bench {mode} {a.variant}: e2e median {out['e2e_ms']['median']:.2f} ms, p95 {out['e2e_ms']['p95']:.2f} ms -> {a.out}")


if __name__ == "__main__":
    main()
