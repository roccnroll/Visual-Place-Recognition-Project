"""
Calcola le recall finali dopo la pipeline adattiva.

Gestisce due gruppi separati:
  - Query rerankate: hanno i .torch del IM top-20 → riordina per num_inliers
  - Query skippate:  usa direttamente il top-1 del retrieval (nessun IM)

Input:
    --preds-dir        cartella con i .txt originali del retrieval
    --reranked-dir     cartella con i .torch del IM top-20 (output di match_queries_preds.py su preds_filtered/)
    --skipped-file     preds_filtered/skipped.txt  (lista query non rerankate)
    --recall-values    es. 1 5 10 20

Output: stampa R@1, R@5, R@10, R@20
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util import read_file_preds, get_utm_from_path, compute_distance


POSITIVE_DIST_THRESHOLD = 25  # metri


def is_positive(pred_path, query_utm, threshold=POSITIVE_DIST_THRESHOLD):
    pred_utm = get_utm_from_path(pred_path)
    return compute_distance(query_utm, pred_utm) <= threshold


def eval_reranked(preds_dir, reranked_dir, recall_values, threshold):
    """Query rerankate: riordina i candidati per num_inliers desc → calcola recall."""
    torch_files = sorted(glob(os.path.join(reranked_dir, "*.torch")),
                         key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else Path(x).stem)

    recalls = np.zeros(len(recall_values))
    count = 0

    for torch_file in tqdm(torch_files, desc="Reranked"):
        qid = Path(torch_file).stem
        txt_file = os.path.join(preds_dir, f"{qid}.txt")
        if not os.path.exists(txt_file):
            continue

        query_path, preds_paths = read_file_preds(txt_file)
        query_utm = get_utm_from_path(query_path)

        results = torch.load(torch_file, map_location="cpu", weights_only=False)
        n_cands = min(len(results), len(preds_paths))

        inliers = torch.tensor([results[i].get("num_inliers", 0) for i in range(n_cands)],
                               dtype=torch.float32)
        order = torch.argsort(inliers, descending=True)
        sorted_preds = [preds_paths[i] for i in order.tolist()]

        for i, k in enumerate(recall_values):
            if any(is_positive(sorted_preds[j], query_utm, threshold)
                   for j in range(min(k, len(sorted_preds)))):
                recalls[i:] += 1
                break

        count += 1

    return recalls, count


def eval_skipped(preds_dir, skipped_ids, recall_values, threshold):
    """Query skippate: usa top-1 del retrieval (nessun IM)."""
    recalls = np.zeros(len(recall_values))
    count = 0

    for qid in tqdm(skipped_ids, desc="Skipped"):
        txt_file = os.path.join(preds_dir, f"{qid}.txt")
        if not os.path.exists(txt_file):
            continue

        query_path, preds_paths = read_file_preds(txt_file)
        query_utm = get_utm_from_path(query_path)

        for i, k in enumerate(recall_values):
            if any(is_positive(preds_paths[j], query_utm, threshold)
                   for j in range(min(k, len(preds_paths)))):
                recalls[i:] += 1
                break

        count += 1

    return recalls, count


def main():
    parser = argparse.ArgumentParser(description="Calcola recall dopo pipeline adattiva")
    parser.add_argument("--preds-dir",    required=True, help="Cartella con i .txt del retrieval originale")
    parser.add_argument("--reranked-dir", required=True, help="Cartella con i .torch del IM top-20")
    parser.add_argument("--skipped-file", required=True, help="File con lista query skippate (una per riga)")
    parser.add_argument("--recall-values", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--positive-dist-threshold", type=int, default=POSITIVE_DIST_THRESHOLD)
    args = parser.parse_args()

    # Carica lista query skippate
    with open(args.skipped_file) as f:
        skipped_ids = [line.strip() for line in f if line.strip()]

    print(f"Query skippate : {len(skipped_ids)}")

    threshold = args.positive_dist_threshold

    recalls_reranked, n_reranked = eval_reranked(
        args.preds_dir, args.reranked_dir, args.recall_values, threshold)

    recalls_skipped, n_skipped = eval_skipped(
        args.preds_dir, skipped_ids, args.recall_values, threshold)

    total = n_reranked + n_skipped
    if total == 0:
        print("Nessuna query trovata.")
        return

    recalls_total = (recalls_reranked + recalls_skipped) / total * 100

    print(f"\nQuery rerankate : {n_reranked}")
    print(f"Query skippate  : {n_skipped}")
    print(f"Totale          : {total}")
    print()
    print("Recall finale (pipeline adattiva):")
    for k, r in zip(args.recall_values, recalls_total):
        print(f"  R@{k}: {r:.1f}")


if __name__ == "__main__":
    main()
