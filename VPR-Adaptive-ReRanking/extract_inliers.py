"""
Estrazione num_inliers via Image Matching top-1.

Esegue match_queries_preds.py --num-preds 1 su tutte le query,
poi legge i .torch risultanti e salva num_inliers in un CSV.

Input:  preds/          (cartella con <query_id>.txt dal retrieval)
Output: match_top1/     (file .torch per query, prodotti da match_queries_preds.py)
        inliers.csv     (query_id, num_inliers)
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
        "--num-preds", "1",
    ]
    print("Eseguo image matching top-1...")
    result = subprocess.run(cmd, check=True)
    return result.returncode


def collect_inliers(match_dir, output_dir):
    torch_files = sorted(glob(os.path.join(match_dir, "*.torch")))
    if not torch_files:
        raise FileNotFoundError(f"Nessun file .torch trovato in {match_dir}")

    rows = []
    for fpath in torch_files:
        query_id = Path(fpath).stem
        data = torch.load(fpath, map_location="cpu")
        # data è una lista di dict, uno per candidato (qui solo top-1)
        num_inliers = data[0].get("num_inliers", 0) if data else 0
        rows.append((query_id, num_inliers))

    out_path = os.path.join(output_dir, "inliers.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "inliers"])
        writer.writerows(rows)

    print(f"Salvato: {out_path}  ({len(rows)} righe)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="IM top-1 su tutte le query → inliers.csv")
    parser.add_argument("--preds-dir",  required=True, help="Cartella con i .txt del retrieval")
    parser.add_argument("--output-dir", required=True, help="Cartella di output (match_top1/ + inliers.csv)")
    parser.add_argument("--matcher",    default="superpoint-lg", help="Metodo di matching (default: superpoint-lg)")
    parser.add_argument("--device",     default="cuda",          help="Dispositivo (default: cuda)")
    parser.add_argument("--im-size",    type=int, default=512,   help="Dimensione immagini (default: 512)")
    args = parser.parse_args()

    match_dir = os.path.join(args.output_dir, "match_top1")
    os.makedirs(match_dir, exist_ok=True)

    run_matching(args.preds_dir, match_dir, args.matcher, args.device, args.im_size)
    collect_inliers(match_dir, args.output_dir)


if __name__ == "__main__":
    main()
