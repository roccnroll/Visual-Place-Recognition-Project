"""
Estrazione SU (Spatial Uncertainty) da z_data.torch.

Input:  z_data.torch  (prodotto da main.py --save_for_uncertainty)
Output: su_scores.csv  (query_id, su_score)

Formula:
    s_i  = 1 / (1 + L2_i)          similarità dalla distanza L2
    RS   = mean(s_1..s_k) / s_0    rank similarity  (s_0 = top-1)
    SD   = median(s) / max(s)       score distribution
    SU   = alpha * RS + (1-alpha) * SD
"""

import argparse
import csv
import os
import torch


def l2_to_su(distances, k=10, alpha=0.5):
    """
    distances: torch array shape (num_queries, num_preds), ordinato per distanza crescente.
    Restituisce (RS, SD, SU) ognuno shape (num_queries,).
    """
    s = 1.0 / (1.0 + distances)          # (Q, P)

    s0 = s[:, 0]                          # top-1 score
    sk = s[:, 1:k+1].mean(dim=1)         # media candidati 2..k+1

    rs = sk / (s0 + 1e-8)
    sd = s.median(dim=1).values / (s.max(dim=1).values + 1e-8)
    su = alpha * rs + (1 - alpha) * sd

    return rs, sd, su


def main():
    parser = argparse.ArgumentParser(description="Estrai SU scores da z_data.torch")
    parser.add_argument("--z-data",    required=True,  help="Percorso a z_data.torch")
    parser.add_argument("--output-dir", required=True, help="Cartella di output (su_scores.csv)")
    parser.add_argument("--k",     type=int,   default=10,  help="Numero candidati per RS (default 10)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Peso RS vs SD (default 0.5)")
    args = parser.parse_args()

    print(f"Carico z_data da: {args.z_data}")
    z_data = torch.load(args.z_data, map_location="cpu")

    # z_data contiene: database_utms, positives_per_query, predictions, distances
    distances = z_data["distances"]   # (num_queries, num_preds)
    if not isinstance(distances, torch.Tensor):
        distances = torch.tensor(distances, dtype=torch.float32)

    print(f"Query: {distances.shape[0]}, Candidati per query: {distances.shape[1]}")

    rs, sd, su = l2_to_su(distances, k=args.k, alpha=args.alpha)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "su_scores.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "RS", "SD", "SU"])
        for i, (r, s, u) in enumerate(zip(rs.tolist(), sd.tolist(), su.tolist())):
            writer.writerow([i, r, s, u])

    print(f"Salvato: {out_path}  ({len(su)} righe, colonne: RS, SD, SU)")


if __name__ == "__main__":
    main()
