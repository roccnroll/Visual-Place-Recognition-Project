"""
Selezione query da rerankare — Scelta 2 della pipeline adattiva.

Legge le feature estratte (su_scores.csv e/o inliers.csv),
applica il regressore logistico dal JSON dei regressori,
e copia i .txt delle query selezionate in preds_filtered/.

Input:
    features/su_scores.csv    (colonne: query_id, RS, SD, SU)
    features/inliers.csv      (colonne: query_id, inliers)
    regressors.json           (formato: model → matchers → matcher → feature_sets → ...)
    preds/                    (predizioni retrieval originali)

Output:
    preds_filtered/           (solo .txt delle query da rerankare)
    preds_filtered/skipped.txt (lista query non rerankate)

Uso:
    python select_queries.py
        --features-dir features/
        --preds-dir preds/
        --regressors-json cosplace_regressors.json
        --matcher sp-lg
        --feature-set SU          # inliers | RS | SD | SU | SU+inliers
        --criterion "P(help)"     # P(hard) | P(help) | P(help)-aP(hurts) | P(help)/P(hurts)>1
        --output-dir preds_filtered/
"""

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path


# --------------------------------------------------------------------------- #
#  Caricamento feature                                                          #
# --------------------------------------------------------------------------- #

def load_csv_as_dict(path, key_col):
    """Restituisce {query_id: {col: value, ...}} per tutte le colonne tranne key_col."""
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row[key_col]
            data[qid] = {k: float(v) for k, v in row.items() if k != key_col}
    return data


def load_features(features_dir, feature_set):
    """
    Ritorna {query_id: {feat_col: value}} con le feature richieste.
    feature_set: inliers | RS | SD | SU | SU+inliers
    """
    feats = {}

    if feature_set in ("RS", "SD", "SU", "SU+inliers"):
        su_path = os.path.join(features_dir, "su_scores.csv")
        su_data = load_csv_as_dict(su_path, "query_id")
        for qid, vals in su_data.items():
            feats.setdefault(qid, {}).update(vals)  # aggiunge RS, SD, SU

    if feature_set in ("inliers", "SU+inliers"):
        inl_path = os.path.join(features_dir, "inliers.csv")
        inl_data = load_csv_as_dict(inl_path, "query_id")
        for qid, vals in inl_data.items():
            feats.setdefault(qid, {}).update(vals)  # aggiunge inliers

    return feats


# --------------------------------------------------------------------------- #
#  Regressore logistico                                                         #
# --------------------------------------------------------------------------- #

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def apply_regressor(reg, feature_values):
    """
    reg: dict con feat_cols, scaler_mean, scaler_scale, coef, intercept
    feature_values: {col: value}
    Restituisce P(classe positiva).
    """
    x = [feature_values[c] for c in reg["feat_cols"]]
    x_scaled = [(xi - mu) / sc for xi, mu, sc
                in zip(x, reg["scaler_mean"], reg["scaler_scale"])]
    logit = reg["intercept"][0] + sum(c * xi for c, xi in zip(reg["coef"][0], x_scaled))
    return sigmoid(logit)


# --------------------------------------------------------------------------- #
#  Criterio di probabilità                                                      #
# --------------------------------------------------------------------------- #

def compute_score(criterion, regressors, feature_values, alpha):
    if criterion == "P(hard)":
        return apply_regressor(regressors["hard"], feature_values)
    elif criterion == "P(help)":
        return apply_regressor(regressors["help"], feature_values)
    elif criterion == "P(help)-aP(hurts)":
        p_help  = apply_regressor(regressors["help"],  feature_values)
        p_hurts = apply_regressor(regressors["hurts"], feature_values)
        return p_help - alpha * p_hurts
    elif criterion == "P(help)/P(hurts)>1":
        p_help  = apply_regressor(regressors["help"],  feature_values)
        p_hurts = apply_regressor(regressors["hurts"], feature_values)
        # score > tau (tau=1.0 fisso) → reranka se P(help)/P(hurts) > 1
        return p_help / (p_hurts + 1e-8)
    else:
        raise ValueError(f"Criterio sconosciuto: {criterion}")


# --------------------------------------------------------------------------- #
#  Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Seleziona query da rerankare")
    parser.add_argument("--features-dir",    required=True)
    parser.add_argument("--preds-dir",       required=True)
    parser.add_argument("--regressors-json", required=True, help="JSON con regressori e soglie")
    parser.add_argument("--matcher",         required=True, help="es. sp-lg, loftr")
    parser.add_argument("--feature-set",     required=True,
                        choices=["inliers", "RS", "SD", "SU", "SU+inliers"])
    parser.add_argument("--criterion",       required=True,
                        choices=["P(hard)", "P(help)", "P(help)-aP(hurts)", "P(help)/P(hurts)>1"])
    parser.add_argument("--output-dir",      required=True)
    args = parser.parse_args()

    # Carica JSON
    with open(args.regressors_json) as f:
        data = json.load(f)

    fs_data    = data["matchers"][args.matcher]["feature_sets"][args.feature_set]
    regressors = fs_data["regressors"]
    hparams    = fs_data["val_hparams"][args.criterion]
    tau        = hparams["tau"]
    alpha      = hparams.get("alpha", 0.5)

    print(f"Modello       : {data['model']}")
    print(f"Matcher       : {args.matcher}")
    print(f"Feature set   : {args.feature_set}")
    print(f"Criterio      : {args.criterion}  |  tau={tau}  |  alpha={alpha}")

    # Carica feature
    feats = load_features(args.features_dir, args.feature_set)
    query_ids = sorted(feats.keys(), key=lambda x: int(x) if x.isdigit() else x)

    # Partiziona
    rerank_ids, skip_ids = [], []
    for qid in query_ids:
        score = compute_score(args.criterion, regressors, feats[qid], alpha)
        (rerank_ids if score > tau else skip_ids).append(qid)

    pct = len(rerank_ids) / len(query_ids) * 100 if query_ids else 0
    print(f"Totale query  : {len(query_ids)}")
    print(f"Da rerankare  : {len(rerank_ids)}  ({pct:.1f}%)")
    print(f"Skippate      : {len(skip_ids)}")

    # Copia .txt delle query da rerankare
    os.makedirs(args.output_dir, exist_ok=True)
    copied, missing = 0, 0
    for qid in rerank_ids:
        src = os.path.join(args.preds_dir, f"{qid}.txt")
        dst = os.path.join(args.output_dir, f"{qid}.txt")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1

    if missing:
        print(f"  WARN: {missing} file .txt non trovati in {args.preds_dir}")

    # Salva lista skippate
    skipped_path = os.path.join(args.output_dir, "skipped.txt")
    with open(skipped_path, "w") as f:
        f.write("\n".join(skip_ids))

    print(f"Copiati {copied} .txt in {args.output_dir}")
    print(f"Skipped salvato in {skipped_path}")


if __name__ == "__main__":
    main()
