"""
Estrazione feature sequenziali via Image Matching top-10.

Esegue match_queries_preds.py --num-preds 10 su tutte le query,
poi legge i .torch risultanti e calcola le feature progressive:
  - num_inliers_top1
  - max_inliers_top5, second_max_inliers_top5, gap_inliers_top5,
    best_retrieval_rank_top5, top1_is_best_top5
  - max_inliers_top10, second_max_inliers_top10, gap_inliers_top10,
    best_retrieval_rank_top10, top1_is_best_top10

Input:  preds/          (predizioni retrieval)
Output: match_top10/    (.torch per query, prodotti da match_queries_preds.py)
        sequential_features.csv
"""

import argparse
import csv
import os
import subprocess
import sys
from glob import glob
from pathlib import Path

import torch


def run_matching(preds_dir, out_dir, matcher, device, im_size):
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "match_queries_preds.py"
    cmd = [
        sys.executable, str(script),
        "--preds-dir", preds_dir,
        "--out-dir",   out_dir,
        "--matcher",   matcher,
        "--device",    device,
        "--im-size",   str(im_size),
        "--num-preds", "10",
    ]
    print("Eseguo image matching top-10...")
    subprocess.run(cmd, check=True)


def prog_features(candidates, budget):
    """Feature progressive per i primi `budget` candidati."""
    sub = [c for c in candidates if c["rank"] <= budget]
    if not sub:
        return {
            f"max_inliers_top{budget}": 0,
            f"second_max_inliers_top{budget}": 0,
            f"gap_inliers_top{budget}": 0,
            f"best_retrieval_rank_top{budget}": budget,
            f"top1_is_best_top{budget}": 0,
        }
    by_inl = sorted(sub, key=lambda c: (-c["inliers"], c["rank"]))
    best = by_inl[0]
    second = by_inl[1]["inliers"] if len(by_inl) >= 2 else 0
    return {
        f"max_inliers_top{budget}":         best["inliers"],
        f"second_max_inliers_top{budget}":  second,
        f"gap_inliers_top{budget}":         best["inliers"] - second,
        f"best_retrieval_rank_top{budget}": best["rank"],
        f"top1_is_best_top{budget}":        int(best["rank"] == 1),
    }


def collect_features(match_dir, output_dir):
    torch_files = sorted(glob(os.path.join(match_dir, "*.torch")))
    if not torch_files:
        raise FileNotFoundError(f"Nessun .torch trovato in {match_dir}")

    rows = []
    for fpath in torch_files:
        query_id = Path(fpath).stem
        data = torch.load(fpath, map_location="cpu", weights_only=False)
        candidates = [{"rank": i + 1, "inliers": d.get("num_inliers", 0)}
                      for i, d in enumerate(data)]
        row = {"query_id": query_id,
               "num_inliers_top1": candidates[0]["inliers"] if candidates else 0}
        row.update(prog_features(candidates, 5))
        row.update(prog_features(candidates, 10))
        rows.append(row)

    cols = [
        "query_id", "num_inliers_top1",
        "max_inliers_top5", "second_max_inliers_top5", "gap_inliers_top5",
        "best_retrieval_rank_top5", "top1_is_best_top5",
        "max_inliers_top10", "second_max_inliers_top10", "gap_inliers_top10",
        "best_retrieval_rank_top10", "top1_is_best_top10",
    ]
    out_path = os.path.join(output_dir, "sequential_features.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Salvato: {out_path}  ({len(rows)} righe)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="IM top-10 su tutte le query → sequential_features.csv")
    parser.add_argument("--preds-dir",  required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--matcher",    default="superpoint-lg")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--im-size",    type=int, default=512)
    args = parser.parse_args()

    match_dir = os.path.join(args.output_dir, "match_top10")
    os.makedirs(match_dir, exist_ok=True)
    run_matching(args.preds_dir, match_dir, args.matcher, args.device, args.im_size)
    collect_features(match_dir, args.output_dir)


if __name__ == "__main__":
    main()
