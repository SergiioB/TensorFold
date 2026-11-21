#!/usr/bin/env python3
"""Card dumps against the host reference's: relative L2 error of every visit row, and the logits' top tokens.
  inkling2_cmp.py REF_DIR CARD_DIR   (files v<visit>_p<pos>.f32 and logits0_p<pos>.f32 from INK2_DUMP)"""
import glob
import os
import re
import sys

import numpy as np


def main():
    ref, card = sys.argv[1], sys.argv[2]
    rows = {}
    worst = 0.0
    for path in sorted(glob.glob(os.path.join(card, "*.f32"))):
        name = os.path.basename(path)
        rp = os.path.join(ref, name)
        if not os.path.exists(rp):
            continue
        a = np.fromfile(path, np.float32)
        b = np.fromfile(rp, np.float32)
        if a.shape != b.shape:
            print("shape differs", name, a.shape, b.shape)
            continue
        m = re.match(r"(v|logits)(\d+)_p(\d+)\.f32", name)
        kind, idx, pos = m.group(1), int(m.group(2)), int(m.group(3))
        if kind == "logits":
            ta, tb = int(np.argmax(a)), int(np.argmax(b))
            top = np.argsort(-b)[:2]
            print(f"logits p{pos}: card top {ta}, host top {tb}, host margin {b[top[0]] - b[top[1]]:.4f}, max |diff| {np.abs(a - b).max():.4f}{'' if ta == tb else '  DIFFERENT'}")
            continue
        rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
        rows.setdefault(idx, []).append((pos, rel))
        worst = max(worst, rel)
    for v in sorted(rows):
        r = rows[v]
        mx = max(x for _, x in r)
        print(f"visit {v:2d}: {len(r)} rows, max rel L2 {mx:.2e} (pos {max(r, key=lambda t: t[1])[0]})")
    print(f"worst visit rel L2 {worst:.2e}")


if __name__ == "__main__":
    main()
