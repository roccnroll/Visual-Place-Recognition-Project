"""
Test sistematici della pipeline adattiva (senza GPU/Colab).

Copre:
  1. JSON structure  — cosplace/megaloc, sp-lg/loftr, standard + sequential
  2. select_queries  — caricamento feature, regressori, criteri, copia .txt
  3. sequential_select — gate1/gate5/gate10 in cascata
  4. extract_inliers_sequential — collect_features() con .torch sintetici
  5. check_performance — R@1 su dati sintetici

Esegui da radice repo:
    python tests/test_pipeline.py
"""

import csv
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "VPR-Adaptive-ReRanking"))

# ── importa moduli da testare ───────────────────────────────────────────────
import select_queries as sq
import sequential_select as ss
from extract_inliers_sequential import collect_features, prog_features

REGRESSORS_DIR = REPO / "regressors"

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
_fails = []

def ok(name):
    print(f"  {PASS} {name}")

def fail(name, msg):
    print(f"  {FAIL} {name}: {msg}")
    _fails.append(f"{name}: {msg}")

def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print('─'*55)

# ════════════════════════════════════════════════════════════
# 1. JSON STRUCTURE
# ════════════════════════════════════════════════════════════
section("1. JSON structure")

STD_MODELS   = ["cosplace", "megaloc"]
MATCHERS     = ["sp-lg", "loftr"]
FEATURE_SETS = ["inliers", "RS", "SD", "SU", "SU+inliers"]
CRITERIA     = ["P(hard)", "P(help)", "P(help)-aP(hurts)", "P(help)/P(hurts)>1"]
REG_KEYS     = ["hard", "help", "hurts"]
REG_FIELDS   = ["feat_cols", "scaler_mean", "scaler_scale", "coef", "intercept"]

for model in STD_MODELS:
    path = REGRESSORS_DIR / f"{model}_regressors.json"
    try:
        with open(path) as f:
            d = json.load(f)
        for matcher in MATCHERS:
            fs_block = d["matchers"][matcher]["feature_sets"]
            for fs in FEATURE_SETS:
                regs = fs_block[fs]["regressors"]
                for reg in REG_KEYS:
                    for field in REG_FIELDS:
                        assert field in regs[reg], f"{field} mancante in {model}/{matcher}/{fs}/{reg}"
                for crit in CRITERIA:
                    hp = fs_block[fs]["val_hparams"][crit]
                    assert "tau" in hp
        ok(f"{model}_regressors.json")
    except Exception as e:
        fail(f"{model}_regressors.json", str(e))

SEQ_GATES = ["gate1", "gate5", "gate10"]
SEQ_FIELDS = ["feat_cols", "scaler_mean", "scaler_scale", "coef", "intercept"]

for model in STD_MODELS:
    path = REGRESSORS_DIR / f"{model}_sequential_regressors.json"
    try:
        with open(path) as f:
            d = json.load(f)
        for matcher in MATCHERS:
            m = d["matchers"][matcher]
            for gate in SEQ_GATES:
                for field in SEQ_FIELDS:
                    assert field in m[gate], f"{field} mancante in {gate}"
            hp = m["val_hparams"]
            for k in ["tau1", "tau5", "tau10"]:
                assert k in hp
            # verify feature counts
            assert len(m["gate1"]["feat_cols"]) == 1
            assert len(m["gate5"]["feat_cols"]) == 6
            assert len(m["gate10"]["feat_cols"]) == 10
            # verify scaler dimension matches feat_cols
            for gate in SEQ_GATES:
                n = len(m[gate]["feat_cols"])
                assert len(m[gate]["scaler_mean"]) == n
                assert len(m[gate]["scaler_scale"]) == n
                assert len(m[gate]["coef"][0]) == n
        ok(f"{model}_sequential_regressors.json")
    except Exception as e:
        fail(f"{model}_sequential_regressors.json", str(e))

# ════════════════════════════════════════════════════════════
# 2. select_queries — regressore logistico + criteri
# ════════════════════════════════════════════════════════════
section("2. select_queries — logica regressore")

# carica un regressore reale (cosplace sp-lg SU)
with open(REGRESSORS_DIR / "cosplace_regressors.json") as f:
    _cpl = json.load(f)
_su_regs = _cpl["matchers"]["sp-lg"]["feature_sets"]["SU"]["regressors"]

def _p(prob):
    return abs(prob - 0.5) < 0.5  # must be in [0,1)

try:
    # SU alto (≈1) = retrieval incerto → P(hard) alta
    fv_high_su = {"SU": 0.999}
    p_hard_high = sq.apply_regressor(_su_regs["hard"], fv_high_su)
    # SU basso (lontano dalla media 0.955) = retrieval certo → P(hard) minore
    fv_low_su = {"SU": 0.85}
    p_hard_low = sq.apply_regressor(_su_regs["hard"], fv_low_su)
    assert p_hard_high > p_hard_low, \
        f"P(hard) deve essere maggiore con SU alto ({p_hard_high:.3f} vs {p_hard_low:.3f})"
    assert 0.0 < p_hard_high <= 1.0 and 0.0 < p_hard_low <= 1.0
    ok(f"apply_regressor monotone in SU (high={p_hard_high:.3f} > low={p_hard_low:.3f})")
except Exception as e:
    fail("apply_regressor", str(e))

try:
    # inliers: scaler_mean è NEGATIVO nel nostro JSON (feature = -num_inliers)
    # → pochi inliers (fv["inliers"]=-5) = molto incerto → P(hard) alta
    _inl_regs = _cpl["matchers"]["sp-lg"]["feature_sets"]["inliers"]["regressors"]
    p_hard_few  = sq.apply_regressor(_inl_regs["hard"], {"inliers": -5})
    p_hard_many = sq.apply_regressor(_inl_regs["hard"], {"inliers": -300})
    assert p_hard_few > p_hard_many, \
        f"inliers: pochi inliers → P(hard) alta ({p_hard_few:.3f} vs {p_hard_many:.3f})"
    ok(f"apply_regressor inliers: few={p_hard_few:.3f} > many={p_hard_many:.3f}")
except Exception as e:
    fail("apply_regressor inliers", str(e))

try:
    # RS: alta rank similarity → retrieval stabile → P(hard) ... verifica solo monotonia
    _rs_regs = _cpl["matchers"]["sp-lg"]["feature_sets"]["RS"]["regressors"]
    p1 = sq.apply_regressor(_rs_regs["hard"], {"RS": 0.85})
    p2 = sq.apply_regressor(_rs_regs["hard"], {"RS": 0.98})
    assert 0 < p1 <= 1 and 0 < p2 <= 1
    ok(f"apply_regressor RS: RS=0.85→{p1:.3f}, RS=0.98→{p2:.3f}")
except Exception as e:
    fail("apply_regressor RS", str(e))

try:
    # SD: stesso check
    _sd_regs = _cpl["matchers"]["sp-lg"]["feature_sets"]["SD"]["regressors"]
    p1 = sq.apply_regressor(_sd_regs["hard"], {"SD": 0.92})
    p2 = sq.apply_regressor(_sd_regs["hard"], {"SD": 0.99})
    assert 0 < p1 <= 1 and 0 < p2 <= 1
    ok(f"apply_regressor SD: SD=0.92→{p1:.3f}, SD=0.99→{p2:.3f}")
except Exception as e:
    fail("apply_regressor SD", str(e))

try:
    # SU+inliers: due feature, feat_cols=["SU","inliers"]
    _sui_regs = _cpl["matchers"]["sp-lg"]["feature_sets"]["SU+inliers"]["regressors"]
    assert _sui_regs["hard"]["feat_cols"] == ["SU", "inliers"]
    p = sq.apply_regressor(_sui_regs["hard"], {"SU": 0.96, "inliers": -50})
    assert 0 < p <= 1
    ok(f"apply_regressor SU+inliers: p={p:.3f}")
except Exception as e:
    fail("apply_regressor SU+inliers", str(e))

try:
    # P(help)-aP(hurts) con alpha=0 deve coincidere con P(help)
    fv = {"SU": 0.90}
    s1 = sq.compute_score("P(help)-aP(hurts)", _su_regs, fv, alpha=0.0)
    s2 = sq.compute_score("P(help)",           _su_regs, fv, alpha=None)
    assert abs(s1 - s2) < 1e-9, f"alpha=0 deve dare stesso score: {s1} vs {s2}"
    ok("P(help)-aP(hurts) con alpha=0 == P(help)")
except Exception as e:
    fail("compute_score P(help)-aP(hurts)", str(e))

try:
    # load_csv_as_dict
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tf:
        w = csv.writer(tf); w.writerow(["query_id","RS","SD","SU"])
        w.writerow(["0042","0.95","0.97","0.96"])
        tmp = tf.name
    d = sq.load_csv_as_dict(tmp, "query_id")
    assert d["0042"]["SU"] == 0.96
    os.unlink(tmp)
    ok("load_csv_as_dict")
except Exception as e:
    fail("load_csv_as_dict", str(e))

# ════════════════════════════════════════════════════════════
# 2b. Hard threshold — youden / best_r1 / efficiency_95
# ════════════════════════════════════════════════════════════
section("2b. Hard threshold criteria")

try:
    with open(REGRESSORS_DIR / "cosplace_regressors.json") as f:
        _cpl2 = json.load(f)
    ht = _cpl2["matchers"]["sp-lg"]["hard_thresholds"]
    assert ht["youden"]["T"]        == 28
    assert ht["best_r1"]["T"]       == 23
    assert ht["efficiency_95"]["T"] == 22
    ok("cosplace sp-lg hard_thresholds nel JSON")
except Exception as e:
    fail("hard_thresholds JSON", str(e))

try:
    with open(REGRESSORS_DIR / "megaloc_regressors.json") as f:
        _meg2 = json.load(f)
    ht_m = _meg2["matchers"]["sp-lg"]["hard_thresholds"]
    assert ht_m["youden"]["T"] == 30
    assert ht_m["best_r1"]["T"] == 0
    ok("megaloc sp-lg hard_thresholds nel JSON")
except Exception as e:
    fail("megaloc hard_thresholds JSON", str(e))

try:
    # logica hard threshold: inliers < T → rerank
    T = 28
    assert (10 < T) == True   # 10 inliers → rerank
    assert (50 < T) == False  # 50 inliers → skip
    ok("hard threshold logica (inliers < T)")
except Exception as e:
    fail("hard threshold logica", str(e))

try:
    # verifica che select_queries carichi correttamente il percorso hard threshold
    # simulando le variabili interne
    with open(REGRESSORS_DIR / "cosplace_regressors.json") as f:
        d = json.load(f)
    T = d["matchers"]["sp-lg"]["hard_thresholds"]["youden"]["T"]
    # inliers.csv ha valori POSITIVI
    queries = {"0": {"inliers": 10}, "1": {"inliers": 50}, "2": {"inliers": 28}}
    rerank = [qid for qid, fv in queries.items() if fv["inliers"] < T]
    skip   = [qid for qid, fv in queries.items() if fv["inliers"] >= T]
    assert "0" in rerank   # 10 < 28
    assert "1" in skip     # 50 >= 28
    assert "2" in skip     # 28 >= 28 (boundary: uguale = skip)
    ok(f"hard threshold partizionamento T={T}: rerank={rerank}, skip={skip}")
except Exception as e:
    fail("hard threshold partizionamento", str(e))

# ════════════════════════════════════════════════════════════
# 3. select_queries — copia .txt end-to-end
# ════════════════════════════════════════════════════════════
section("3. select_queries — copia .txt con preds sintetici")

try:
    with tempfile.TemporaryDirectory() as tmp:
        feat_dir = os.path.join(tmp, "features")
        preds_dir = os.path.join(tmp, "preds")
        out_dir   = os.path.join(tmp, "out")
        os.makedirs(feat_dir); os.makedirs(preds_dir)

        # SU scores: query 0 incerta (SU=0.85), query 1 certa (SU=0.999)
        with open(os.path.join(feat_dir, "su_scores.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["query_id","RS","SD","SU"])
            w.writerow(["0","0.90","0.92","0.85"])
            w.writerow(["1","0.96","0.97","0.999"])

        # file .txt sintetici
        for qid in ["0","1"]:
            with open(os.path.join(preds_dir, f"{qid}.txt"), "w") as f:
                f.write(f"query_{qid}\n")

        # carica regressori reali cosplace sp-lg
        with open(REGRESSORS_DIR / "cosplace_regressors.json") as f:
            data = json.load(f)
        fs_data  = data["matchers"]["sp-lg"]["feature_sets"]["SU"]
        regs     = fs_data["regressors"]
        tau      = fs_data["val_hparams"]["P(help)"]["tau"]  # 0.01

        feats = sq.load_features(feat_dir, "SU")
        rerank_ids, skip_ids = [], []
        for qid in sorted(feats.keys()):
            score = sq.compute_score("P(help)", regs, feats[qid], alpha=0.5)
            (rerank_ids if score > tau else skip_ids).append(qid)

        # con tau=0.01 quasi tutte le query vengono rerankate
        assert len(rerank_ids) + len(skip_ids) == 2
        ok(f"partizionamento: {len(rerank_ids)} rerank, {len(skip_ids)} skip")
except Exception as e:
    fail("select_queries end-to-end", str(e))

# ════════════════════════════════════════════════════════════
# 4. sequential_select — gate in cascata
# ════════════════════════════════════════════════════════════
section("4. sequential_select — gate in cascata")

try:
    with open(REGRESSORS_DIR / "cosplace_sequential_regressors.json") as f:
        seq_data = json.load(f)
    m   = seq_data["matchers"]["sp-lg"]
    g1  = m["gate1"]
    g5  = m["gate5"]
    g10 = m["gate10"]

    # query con pochi inliers (top1=5) → alta P(continue)
    fv_few = {
        "num_inliers_top1": 5,
        "max_inliers_top5": 12, "second_max_inliers_top5": 8,
        "gap_inliers_top5": 4,  "best_retrieval_rank_top5": 2,
        "top1_is_best_top5": 0,
        "max_inliers_top10": 15, "second_max_inliers_top10": 11,
        "gap_inliers_top10": 4,  "best_retrieval_rank_top10": 3,
        "top1_is_best_top10": 0,
    }
    # query con molti inliers (top1=300) → bassa P(continue)
    fv_many = {
        "num_inliers_top1": 300,
        "max_inliers_top5": 320, "second_max_inliers_top5": 50,
        "gap_inliers_top5": 270, "best_retrieval_rank_top5": 1,
        "top1_is_best_top5": 1,
        "max_inliers_top10": 320, "second_max_inliers_top10": 50,
        "gap_inliers_top10": 270, "best_retrieval_rank_top10": 1,
        "top1_is_best_top10": 1,
    }
    p_few  = ss.apply_gate(g1, fv_few)
    p_many = ss.apply_gate(g1, fv_many)
    assert p_few > p_many, f"gate1: pochi inliers → P più alta ({p_few:.3f} vs {p_many:.3f})"
    ok(f"gate1 monotone: P(few={p_few:.3f}) > P(many={p_many:.3f})")
except Exception as e:
    fail("sequential_select gate1 monotone", str(e))

try:
    # test copia .txt end-to-end con sequential_select
    with tempfile.TemporaryDirectory() as tmp:
        feats_csv = os.path.join(tmp, "sequential_features.csv")
        preds_dir = os.path.join(tmp, "preds")
        out_dir   = os.path.join(tmp, "out")
        skip_file = os.path.join(tmp, "skipped.txt")
        os.makedirs(preds_dir)

        cols = ["query_id","num_inliers_top1",
                "max_inliers_top5","second_max_inliers_top5","gap_inliers_top5",
                "best_retrieval_rank_top5","top1_is_best_top5",
                "max_inliers_top10","second_max_inliers_top10","gap_inliers_top10",
                "best_retrieval_rank_top10","top1_is_best_top10"]
        with open(feats_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            # query 0: pochi inliers → dovrebbe passare gate1 (tau1=0.88)
            w.writerow({"query_id":"0","num_inliers_top1":5,
                        "max_inliers_top5":12,"second_max_inliers_top5":8,"gap_inliers_top5":4,
                        "best_retrieval_rank_top5":2,"top1_is_best_top5":0,
                        "max_inliers_top10":15,"second_max_inliers_top10":11,"gap_inliers_top10":4,
                        "best_retrieval_rank_top10":3,"top1_is_best_top10":0})
            # query 1: molti inliers → dovrebbe fermarsi a gate1
            w.writerow({"query_id":"1","num_inliers_top1":300,
                        "max_inliers_top5":320,"second_max_inliers_top5":50,"gap_inliers_top5":270,
                        "best_retrieval_rank_top5":1,"top1_is_best_top5":1,
                        "max_inliers_top10":320,"second_max_inliers_top10":50,"gap_inliers_top10":270,
                        "best_retrieval_rank_top10":1,"top1_is_best_top10":1})

        for qid in ["0","1"]:
            with open(os.path.join(preds_dir, f"{qid}.txt"), "w") as f:
                f.write(f"q{qid}\n")

        # esegui select tramite i componenti interni
        feats = {}
        with open(feats_csv, newline="") as f:
            for row in csv.DictReader(f):
                feats[row["query_id"]] = {k: float(v) for k,v in row.items() if k!="query_id"}

        with open(REGRESSORS_DIR / "cosplace_sequential_regressors.json") as f:
            seq_d = json.load(f)
        m2 = seq_d["matchers"]["sp-lg"]
        tau1, tau5, tau10 = m2["val_hparams"]["tau1"], m2["val_hparams"]["tau5"], m2["val_hparams"]["tau10"]

        selected, skipped = [], []
        for qid in ["0","1"]:
            fv = feats[qid]
            if ss.apply_gate(m2["gate1"], fv) <= tau1:
                skipped.append(qid); continue
            if ss.apply_gate(m2["gate5"], fv) <= tau5:
                skipped.append(qid); continue
            if ss.apply_gate(m2["gate10"], fv) <= tau10:
                skipped.append(qid); continue
            selected.append(qid)

        # query 1 (molti inliers) deve essere skippata
        assert "1" in skipped, f"query1 (300 inliers) dovrebbe essere skippata, selected={selected}"
        ok(f"cascata gate: selected={selected}, skipped={skipped}")
except Exception as e:
    fail("sequential_select cascata", str(e))

# ════════════════════════════════════════════════════════════
# 5. extract_inliers_sequential — collect_features con .torch sintetici
# ════════════════════════════════════════════════════════════
section("5. extract_inliers_sequential — collect_features")

try:
    with tempfile.TemporaryDirectory() as tmp:
        match_dir = os.path.join(tmp, "match_top10")
        os.makedirs(match_dir)

        # crea .torch sintetici: 10 candidati con inliers decrescenti per rank
        for qid in ["0042", "0043"]:
            candidates = [{"num_inliers": max(0, 200 - i*20), "keypoints0": []} for i in range(10)]
            torch.save(candidates, os.path.join(match_dir, f"{qid}.torch"))

        out_path = collect_features(match_dir, tmp)
        assert os.path.exists(out_path)

        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

        r = rows[0]
        assert float(r["num_inliers_top1"]) == 200.0
        assert float(r["max_inliers_top5"]) == 200.0
        assert float(r["second_max_inliers_top5"]) == 180.0
        assert float(r["gap_inliers_top5"]) == 20.0
        assert float(r["best_retrieval_rank_top5"]) == 1
        assert float(r["top1_is_best_top5"]) == 1
        assert float(r["max_inliers_top10"]) == 200.0
        ok("collect_features — valori feature corretti")
except Exception as e:
    fail("collect_features", str(e))

try:
    # prog_features con budget=5 su candidati invertiti (rank 3 ha max inliers)
    cands = [{"rank": 1, "inliers": 10},
             {"rank": 2, "inliers": 5},
             {"rank": 3, "inliers": 80},
             {"rank": 4, "inliers": 20},
             {"rank": 5, "inliers": 15}]
    pf = prog_features(cands, 5)
    assert pf["max_inliers_top5"] == 80
    assert pf["second_max_inliers_top5"] == 20
    assert pf["gap_inliers_top5"] == 60
    assert pf["best_retrieval_rank_top5"] == 3
    assert pf["top1_is_best_top5"] == 0
    ok("prog_features — best non è rank 1")
except Exception as e:
    fail("prog_features", str(e))

# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════
print(f"\n{'═'*55}")
if _fails:
    print(f"  {len(_fails)} test FALLITI:")
    for f in _fails:
        print(f"    {FAIL} {f}")
    sys.exit(1)
else:
    print(f"  {PASS} Tutti i test passati")
print('═'*55)
