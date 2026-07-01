"""
Selezione query per pipeline sequenziale — tre gate in cascata.

  gate1  (num_inliers_top1)          → stop | continua
  gate5  (6 feature, top1+top5)      → stop | continua
  gate10 (10 feature, top1+top5+top10) → stop | selezionata per IM top-20

Le query che non superano tutti e tre i gate vengono skippate (retrieval top-1).
Le query selezionate ricevono IM top-20 (step successivo nella pipeline).

Input:
    sequential_features.csv         (da extract_inliers_sequential.py)
    {model}_sequential_regressors.json
    preds/                           (predizioni retrieval)

Output:
    preds_filtered/                  (.txt delle query selezionate)
    skipped.txt                      (query non selezionate)

Uso:
    python sequential_select.py
        --features-csv features/sequential_features.csv
        --preds-dir preds/
        --regressors-json regressors/cosplace_sequential_regressors.json
        --matcher superpoint-lg
        --output-dir preds_filtered/
        --skipped-file skipped.txt
"""

import argparse
import csv
import json
import math
import os
import shutil


MATCHER_ALIAS = {"superpoint-lg": "sp-lg", "superpoint_lg": "sp-lg"}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def apply_gate(gate, fv):
    x = [fv[c] for c in gate["feat_cols"]]
    xs = [(xi - mu) / sc for xi, mu, sc
          in zip(x, gate["scaler_mean"], gate["scaler_scale"])]
    logit = gate["intercept"][0] + sum(c * xi for c, xi in zip(gate["coef"][0], xs))
    return sigmoid(logit)


def main():
    parser = argparse.ArgumentParser(
        description="Selezione sequenziale (gate1→gate5→gate10) → preds_filtered/")
    parser.add_argument("--features-csv",    required=True,
                        help="sequential_features.csv da extract_inliers_sequential.py")
    parser.add_argument("--preds-dir",       required=True)
    parser.add_argument("--regressors-json", required=True)
    parser.add_argument("--matcher",         required=True)
    parser.add_argument("--output-dir",      required=True)
    parser.add_argument("--skipped-file",    default=None)
    args = parser.parse_args()

    matcher_key = MATCHER_ALIAS.get(args.matcher, args.matcher)

    with open(args.regressors_json) as f:
        data = json.load(f)
    m = data["matchers"][matcher_key]
    gate1  = m["gate1"]
    gate5  = m["gate5"]
    gate10 = m["gate10"]
    hp     = m["val_hparams"]
    tau1, tau5, tau10 = hp["tau1"], hp["tau5"], hp["tau10"]

    print(f"Modello : {data['model']}  Matcher: {matcher_key}")
    print(f"Soglie  : tau1={tau1}  tau5={tau5}  tau10={tau10}")
    if "note" in hp:
        print(f"  NOTA: {hp['note']}")

    feats = {}
    with open(args.features_csv, newline="") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"]
            feats[qid] = {k: float(v) for k, v in row.items() if k != "query_id"}

    query_ids = sorted(feats.keys(), key=lambda x: int(x) if x.isdigit() else x)

    selected, skipped = [], []
    n1 = n5 = n10 = 0

    for qid in query_ids:
        fv = feats[qid]
        if apply_gate(gate1, fv) <= tau1:
            n1 += 1; skipped.append(qid); continue
        if apply_gate(gate5, fv) <= tau5:
            n5 += 1; skipped.append(qid); continue
        if apply_gate(gate10, fv) <= tau10:
            n10 += 1; skipped.append(qid); continue
        selected.append(qid)

    total = len(query_ids)
    pct = lambda n: f"{n/total*100:.1f}%" if total else "—"
    print(f"\nTotale query  : {total}")
    print(f"Fermate gate1 : {n1}  ({pct(n1)})")
    print(f"Fermate gate5 : {n5}  ({pct(n5)})")
    print(f"Fermate gate10: {n10}  ({pct(n10)})")
    print(f"Selezionate   : {len(selected)}  ({pct(len(selected))})")

    os.makedirs(args.output_dir, exist_ok=True)
    copied = missing = 0
    for qid in selected:
        src = os.path.join(args.preds_dir, f"{qid}.txt")
        dst = os.path.join(args.output_dir, f"{qid}.txt")
        if os.path.exists(src):
            shutil.copy2(src, dst); copied += 1
        else:
            missing += 1
    if missing:
        print(f"  WARN: {missing} .txt non trovati in {args.preds_dir}")

    if args.skipped_file:
        skipped_path = args.skipped_file
    else:
        skipped_path = os.path.join(
            os.path.dirname(args.output_dir.rstrip("/\\")), "skipped.txt")
    with open(skipped_path, "w") as f:
        f.write("\n".join(skipped))

    print(f"\nCopiati {copied} .txt in {args.output_dir}")
    print(f"Skipped salvato in {skipped_path}")


if __name__ == "__main__":
    main()
