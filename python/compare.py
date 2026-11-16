"""Compare two detection dumps (C++ vs Python reference) box by box.

A C++ box matches a Python box when the class is the same and IoU >= --iou-thr (default 0.9); each box is used at most
once (greedy, highest IoU first). Agreement = matched / max(n_cpp, n_py), so extra or missing boxes on either side count against it.
Exit code 1 if overall agreement of any compared group is below --min-agreement.
"""
import argparse
import json
import sys
from pathlib import Path


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(cpp, py, thr):
    """Greedy one-to-one matching. Returns list of (i_cpp, i_py, iou)."""
    pairs = sorted(((iou(c, p), i, j) for i, c in enumerate(cpp) for j, p in enumerate(py)
                    if c[5] == p[5] and iou(c, p) >= thr), reverse=True)
    used_c, used_p, out = set(), set(), []
    for v, i, j in pairs:
        if i not in used_c and j not in used_p:
            used_c.add(i); used_p.add(j); out.append((i, j, v))
    return out


def compare(cpp_dump, py_dump, thr):
    c, p = json.loads(Path(cpp_dump).read_text()), json.loads(Path(py_dump).read_text())
    assert c["images"].keys() == p["images"].keys(), "the two dumps cover different images"
    res = {"meta": {"cpp": c["meta"], "python": p["meta"], "iou_thr": thr}, "groups": {}}
    for group in ("persons", "items"):
        per_image, tot = {}, dict(matched=0, matched_iou50=0, n_cpp=0, n_py=0)
        min_iou, max_ds = 1.0, 0.0
        for name in sorted(c["images"]):
            a, b = c["images"][name][group], p["images"][name][group]
            m = match(a, b, thr)
            m50 = match(a, b, 0.5)
            for i, j, v in m:
                min_iou = min(min_iou, v)
                max_ds = max(max_ds, abs(a[i][4] - b[j][4]))
            per_image[name] = {"matched": len(m), "n_cpp": len(a), "n_py": len(b), "matched_iou50": len(m50)}
            tot["matched"] += len(m); tot["matched_iou50"] += len(m50); tot["n_cpp"] += len(a); tot["n_py"] += len(b)
        denom = max(tot["n_cpp"], tot["n_py"])
        res["groups"][group] = {
            **tot, "agreement_pct": 100.0 * tot["matched"] / denom if denom else 100.0,
            "min_matched_iou": min_iou, "max_matched_score_diff": max_ds,
            "images_with_any_disagreement": sum(1 for v in per_image.values() if v["matched"] != max(v["n_cpp"], v["n_py"])),
            "per_image": per_image}
    res["input_pixel_mismatch_images"] = sum(1 for n in c["images"] if c["images"][n]["pixel_sum"] != p["images"][n]["pixel_sum"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cpp"); ap.add_argument("python"); ap.add_argument("--out", required=True)
    ap.add_argument("--iou-thr", type=float, default=0.9)
    ap.add_argument("--min-agreement", type=float, default=95.0)
    a = ap.parse_args()
    res = compare(a.cpp, a.python, a.iou_thr)
    res["min_agreement_pct"] = a.min_agreement
    ok = all(g["agreement_pct"] >= a.min_agreement for g in res["groups"].values() if g["n_cpp"] or g["n_py"])
    res["passed"] = ok
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k, g in res["groups"].items():
        print(f"  {k:8s} matched {g['matched']}/{max(g['n_cpp'], g['n_py'])} (cpp {g['n_cpp']}, py {g['n_py']}) "
              f"= {g['agreement_pct']:.2f}%  min IoU {g['min_matched_iou']:.4f}  max score diff {g['max_matched_score_diff']:.2e}")
    print(f"  input pixel mismatches: {res['input_pixel_mismatch_images']} images; {'PASS' if ok else 'FAIL'} (threshold {a.min_agreement}%)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
