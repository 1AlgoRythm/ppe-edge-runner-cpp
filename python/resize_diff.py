"""How different are the letterboxed pixels of the two OpenCV builds? (C++ runner: Homebrew OpenCV; Python: opencv-python wheel.)
usage: resize_diff.py LB_DIR OUT.json   - LB_DIR holds <image>_person.png written by `ppe_runner --save-letterbox`."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ref

lb_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = {}
for p in sorted(Path("data/person_test").glob("*.jpg")):
    mine, *_ = ref.letterbox(cv2.imread(str(p)))
    theirs = cv2.imread(str(lb_dir / f"{p.stem}_person.png"))
    d = np.abs(mine.astype(int) - theirs.astype(int))
    rows[p.name] = {"frac_values_differing": float((d > 0).mean()), "max_abs_diff": int(d.max())}
res = {"opencv_python": cv2.__version__, "n_images": len(rows),
       "images_with_any_difference": sum(1 for r in rows.values() if r["max_abs_diff"] > 0),
       "max_abs_diff_overall": max(r["max_abs_diff"] for r in rows.values()),
       "mean_frac_values_differing": float(np.mean([r["frac_values_differing"] for r in rows.values()])),
       "per_image": rows}
out.write_text(json.dumps(res, indent=1))
print(f"letterbox pixels: {res['images_with_any_difference']}/{res['n_images']} images differ, max |diff| {res['max_abs_diff_overall']}, "
      f"{100 * res['mean_frac_values_differing']:.1f}% of values differ on average")
