#!/usr/bin/env python3
# See previous cell for full content; re-inserting due to session reset.
from __future__ import annotations
import argparse, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import numpy as np, pandas as pd

WHITE_MAX = 69
PB_MAX = 26

DEFAULT_BANDS = {
    "pos1": (4, 17),
    "pos2": (12, 33),
    "pos3": (21, 49),
    "pos4": (33, 60),
    "pos5": (51, 66),
    "pb":   (4, 22),
}

LOOKBACK_YEARS = 6
RECENCY_HALF_LIFE_DRAWS = 45.0
RECENCY_STRENGTH_WHITE = 0.35
RECENCY_STRENGTH_PB = 0.35
PRIOR_STRENGTH_WHITE = 120.0
PRIOR_STRENGTH_PB = 26.0
REPEAT_PENALTY_WHITE = 0.65
OVERALL_SCORE_STYLE = "geomean"

POSITION_WEIGHTS = {"pos1":1.0,"pos2":1.0,"pos3":1.0,"pos4":1.0,"pos5":1.0,"pb":0.8}

def load_draws(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = ["date","pos1","pos2","pos3","pos4","pos5","pb"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    for c in ["pos1","pos2","pos3","pos4","pos5","pb"]:
        df[c] = df[c].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df

def beta_pdf(a: float, b: float, x: float) -> float:
    if x <= 0.0 or x >= 1.0: return 0.0
    logB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return math.exp((a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x) - logB)

def order_statistic_prior_white(pos_key: str, prior_strength: float) -> np.ndarray:
    k = int(pos_key[-1])
    a, b = float(k), float(6 - k)
    xs = (np.arange(1, WHITE_MAX + 1) - 0.5) / WHITE_MAX
    pdf = np.array([beta_pdf(a, b, float(x)) for x in xs], dtype=float)
    pdf = np.clip(pdf, 1e-12, None)
    prior = pdf / pdf.sum() * prior_strength
    return prior.astype(float)

def prior_pb_uniform(prior_strength: float) -> np.ndarray:
    return np.ones(PB_MAX, dtype=float) * (prior_strength / PB_MAX)

def normalize(a: np.ndarray) -> np.ndarray:
    s = float(a.sum())
    return a / s if s > 0 else np.zeros_like(a, dtype=float)

def build_posterior_counts(df_hist: pd.DataFrame, prior_white_strength: float, prior_pb_strength: float) -> Dict[str, np.ndarray]:
    post: Dict[str, np.ndarray] = {}
    for i in range(1,6):
        pos = f"pos{i}"
        counts = df_hist[pos].value_counts().reindex(range(1, WHITE_MAX+1), fill_value=0).sort_index().values.astype(float)
        post[pos] = counts + order_statistic_prior_white(pos, prior_white_strength)
    counts_pb = df_hist["pb"].value_counts().reindex(range(1, PB_MAX+1), fill_value=0).sort_index().values.astype(float)
    post["pb"] = counts_pb + prior_pb_uniform(prior_pb_strength)
    return post

def compute_exponential_recency(df_hist: pd.DataFrame, half_life_draws: float,
                                strength_white: float, strength_pb: float):
    if df_hist.empty:
        return np.ones(WHITE_MAX, dtype=float), np.ones(PB_MAX, dtype=float)
    n = len(df_hist)
    ages = (n - 1) - np.arange(n)
    decay = np.power(0.5, ages / float(half_life_draws))
    rec_white = np.zeros(WHITE_MAX, dtype=float)
    for i, row in enumerate(df_hist.itertuples(index=False)):
        w = float(decay[i])
        rec_white[row.pos1 - 1] += w
        rec_white[row.pos2 - 1] += w
        rec_white[row.pos3 - 1] += w
        rec_white[row.pos4 - 1] += w
        rec_white[row.pos5 - 1] += w
    rec_pb = np.zeros(PB_MAX, dtype=float)
    for i, row in enumerate(df_hist.itertuples(index=False)):
        rec_pb[row.pb - 1] += float(decay[i])
    mult_white = 1.0 + strength_white * rec_white
    mult_pb = 1.0 + strength_pb * rec_pb
    return mult_white, mult_pb

def build_weights_for_position(
    pos_key, posterior_counts, band, recency_mult_white, recency_mult_pb,
    used_white_vals, repeat_penalty, no_repeat_penalty, prev_val=None,
    guide_mult_white=None, guide_mult_pb=None
):
    is_pb = (pos_key == "pb")
    lo, hi = band
    start = lo
    if prev_val is not None and not is_pb:
        start = max(start, prev_val + 1)
    if start > hi:
        return np.array([], dtype=int), np.array([], dtype=float)
    candidates = np.arange(start, hi + 1, dtype=int)
    base = posterior_counts[pos_key].copy()
    base = base * (recency_mult_pb if is_pb else recency_mult_white)
    if not is_pb and guide_mult_white is not None and pos_key in guide_mult_white:
        gm = guide_mult_white[pos_key]
        if gm.shape[0] == base.shape[0]: base = base * gm
    if is_pb and guide_mult_pb is not None and guide_mult_pb.shape[0] == base.shape[0]:
        base = base * guide_mult_pb
    weights = base[candidates - 1].astype(float)
    if not is_pb and not no_repeat_penalty and repeat_penalty != 1.0:
        repeat_mask = np.isin(candidates, list(used_white_vals))
        weights[repeat_mask] *= repeat_penalty
    if weights.sum() <= 0:
        weights = np.ones_like(weights, dtype=float)
    s = float(weights.sum())
    return candidates, weights/s

def argmax_pick(cand, p): return int(cand[int(np.argmax(p))])
def confidence_for_choice(candidates, probs, choice):
    if candidates.size == 0: return 0.0
    idx = np.where(candidates == choice)[0]
    return float(probs[idx[0]]) if idx.size > 0 else 0.0

def overall_score(conf_pos: Dict[str, float], style: str) -> float:
    keys = ["pos1","pos2","pos3","pos4","pos5","pb"]
    vals = np.array([conf_pos[k] for k in keys], dtype=float)
    if style == "mean":
        return float(np.mean(vals))
    v = np.where(vals > 0, vals, 1e-12)
    return float(np.exp(np.mean(np.log(v))))

@dataclass
class Ticket:
    pos1:int; pos2:int; pos3:int; pos4:int; pos5:int; pb:int
    conf_pos: Dict[str,float]; conf_overall: float
    def as_tuple(self): return (self.pos1,self.pos2,self.pos3,self.pos4,self.pos5,self.pb)

@dataclass
class BeamItem:
    whites: Tuple[int,int,int,int,int]
    logp: float

def build_probs_for_position(pos_key, posterior_counts, band, recency_mult_white, recency_mult_pb, prev_val=None):
    is_pb = (pos_key == "pb")
    lo, hi = band
    start = lo
    if prev_val is not None and not is_pb:
        start = max(start, prev_val + 1)
    if start > hi: return np.array([], dtype=int), np.array([], dtype=float)
    candidates = np.arange(start, hi + 1, dtype=int)
    base = posterior_counts[pos_key].copy()
    base = base * (recency_mult_pb if is_pb else recency_mult_white)
    probs = base[candidates - 1].astype(float)
    s = probs.sum()
    return candidates, (probs/s if s>0 else np.zeros_like(probs))

def top_k_white_sequences(posterior, bands, rec_mult_white, K, beam):
    cand1, p1 = build_probs_for_position("pos1", posterior, bands["pos1"], rec_mult_white, np.ones(PB_MAX), None)
    beams = [BeamItem((int(cand1[i]),0,0,0,0), float(np.log(max(p1[i],1e-18)))) for i in range(len(cand1))]
    beams.sort(key=lambda x: x.logp, reverse=True); beams = beams[:beam]
    def extend(beams_in, pos_key, idx):
        out = []
        for b in beams_in:
            prev = b.whites[idx-2] if idx>1 else None
            cand, p = build_probs_for_position(pos_key, posterior, bands[pos_key], rec_mult_white, np.ones(PB_MAX), prev)
            for i in range(len(cand)):
                val = int(cand[i])
                if prev is not None and val <= prev: continue
                logp = b.logp + float(np.log(max(p[i], 1e-18)))
                if idx==2: whites=(b.whites[0],val,0,0,0)
                elif idx==3: whites=(b.whites[0],b.whites[1],val,0,0)
                elif idx==4: whites=(b.whites[0],b.whites[1],b.whites[2],val,0)
                else: whites=(b.whites[0],b.whites[1],b.whites[2],b.whites[3],val)
                out.append(BeamItem(whites, logp))
        out.sort(key=lambda x: x.logp, reverse=True)
        return out[:beam]
    beams = extend(beams, "pos2", 2)
    beams = extend(beams, "pos3", 3)
    beams = extend(beams, "pos4", 4)
    beams = extend(beams, "pos5", 5)
    beams.sort(key=lambda x: x.logp, reverse=True)
    return beams[:max(1, K*5)]

def top_pb_choices(posterior, band_pb, rec_mult_pb, top_n):
    cand, p = build_probs_for_position("pb", posterior, band_pb, np.ones(WHITE_MAX), rec_mult_pb, None)
    items = [(int(cand[i]), float(p[i])) for i in range(len(cand))]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:max(1, top_n)]

def compute_guidance_multipliers(post, bands, rec_mult_white, rec_mult_pb, guide_pool, beam, guide_strength_white, guide_strength_pb):
    whites_pool = top_k_white_sequences(post, bands, rec_mult_white, guide_pool, beam)
    if not whites_pool:
        gm_white = {f"pos{i}": np.ones(WHITE_MAX, dtype=float) for i in range(1,6)}
        gm_pb = np.ones(PB_MAX, dtype=float)
        return gm_white, gm_pb
    lp = np.array([w.logp for w in whites_pool], dtype=float); lp -= lp.max(); wts = np.exp(lp); wts = wts / wts.sum()
    gm_white = {f"pos{i}": np.zeros(WHITE_MAX, dtype=float) for i in range(1,6)}
    for w, wt in zip(whites_pool, wts):
        vals = list(w.whites)
        for i, val in enumerate(vals, start=1):
            if 1 <= val <= WHITE_MAX:
                gm_white[f"pos{i}"][val-1] += wt
    for k in gm_white.keys():
        prob = gm_white[k]
        if prob.sum() <= 0: gm_white[k] = np.ones(WHITE_MAX, dtype=float); continue
        prob = prob / prob.sum()
        mean = 1.0 / WHITE_MAX
        ratio = np.where(prob>0, prob/mean, 0.0)
        gm_white[k] = (1.0 - guide_strength_white) + guide_strength_white * ratio
    cand_pb, p_pb = build_probs_for_position("pb", post, bands["pb"], np.ones(WHITE_MAX), rec_mult_pb, None)
    gm_pb = np.ones(PB_MAX, dtype=float)
    if cand_pb.size > 0 and p_pb.sum()>0:
        probs = p_pb / p_pb.sum(); mean = 1.0 / PB_MAX; ratio = probs/mean
        gm_pb[cand_pb-1] = (1.0 - guide_strength_pb) + guide_strength_pb * ratio
    return gm_white, gm_pb

def deterministic_generate(df, bands, n_sets, lookback_years, prior_white, prior_pb, rec_half_life, rec_str_white, rec_str_pb, beam, overlap_penalty, top_pb_n):
    end_date = df["date"].max()
    df_hist = df[(df["date"] <= end_date) & (df["date"] >= end_date - pd.DateOffset(years=lookback_years))].copy()
    post = build_posterior_counts(df_hist, prior_white, prior_pb)
    rec_mult_white, rec_mult_pb = compute_exponential_recency(df_hist, rec_half_life, rec_str_white, rec_str_pb)
    whites_pool = top_k_white_sequences(post, bands, rec_mult_white, n_sets, beam)
    pb_top = top_pb_choices(post, bands["pb"], rec_mult_pb, top_pb_n)
    candidates = []
    for w in whites_pool:
        confs = {}; prev = None
        for idx, key in enumerate(["pos1","pos2","pos3","pos4","pos5"]):
            cand, p = build_probs_for_position(key, post, bands[key], rec_mult_white, rec_mult_pb, prev)
            chosen = w.whites[idx]
            j = np.where(cand == chosen)[0]
            confs[key] = float(p[j[0]]) if j.size > 0 else 0.0
            prev = chosen
        for pb_val, pb_p in pb_top:
            cand_pb, p_pb = build_probs_for_position("pb", post, bands["pb"], rec_mult_white, rec_mult_pb, None)
            j = np.where(cand_pb == pb_val)[0]
            confs2 = dict(confs)
            confs2["pb"] = float(p_pb[j[0]]) if j.size > 0 else 0.0
            overall = overall_score(confs2, OVERALL_SCORE_STYLE)
            base_logp = float(w.logp + math.log(max(confs2["pb"], 1e-18)))
            t = Ticket(w.whites[0], w.whites[1], w.whites[2], w.whites[3], w.whites[4], pb_val, confs2, overall)
            t.base_logp = base_logp  # type: ignore
            candidates.append(t)
    candidates.sort(key=lambda t: (t.base_logp, t.pos1, t.pos2, t.pos3, t.pos4, t.pos5, t.pb), reverse=True)
    selected = []; used_sets = []
    while len(selected) < n_sets and candidates:
        best = None; best_score = -1e18
        for t in candidates:
            overlap = 0
            for s in used_sets:
                overlap += len(s & set([t.pos1,t.pos2,t.pos3,t.pos4,t.pos5]))
            score = t.base_logp - 0.15 * float(overlap)
            if score > best_score: best = t; best_score = score
        if best is None: break
        selected.append(best)
        used_sets.append(set([best.pos1,best.pos2,best.pos3,best.pos4,best.pos5]))
        candidates = [c for c in candidates if (c.pos1,c.pos2,c.pos3,c.pos4,c.pos5,c.pb)!=(best.pos1,best.pos2,best.pos3,best.pos4,best.pos5,best.pb)]
    return selected

def generate_tickets_hybrid_det(df, bands, n_sets, lookback_years, prior_white, prior_pb, rec_half_life, rec_str_white, rec_str_pb, repeat_penalty_white, no_repeat_penalty, seed, guide_pool, beam, guide_strength_white, guide_strength_pb):
    rng = np.random.default_rng(seed)
    end_date = df["date"].max()
    df_hist = df[(df["date"] <= end_date) & (df["date"] >= end_date - pd.DateOffset(years=lookback_years))].copy()
    post = build_posterior_counts(df_hist, prior_white, prior_pb)
    rec_mult_white, rec_mult_pb = compute_exponential_recency(df_hist, rec_half_life, rec_str_white, rec_str_pb)
    gm_white, gm_pb = compute_guidance_multipliers(post, bands, rec_mult_white, rec_mult_pb, guide_pool, beam, guide_strength_white, guide_strength_pb)
    results = []; used_white_vals: Set[int] = set()
    def pick_one(use_greedy, prev_val):
        chosen = {}; confs = {}; pval = prev_val
        for key in ["pos1","pos2","pos3","pos4","pos5"]:
            cand, probs = build_weights_for_position(key, post, bands[key], rec_mult_white, rec_mult_pb, used_white_vals, repeat_penalty_white, no_repeat_penalty, pval, gm_white, None)
            if cand.size == 0: return {}, {}
            pick = int(cand[int(np.argmax(probs))]) if use_greedy else int(rng.choice(cand, p=probs))
            chosen[key] = pick
            confs[key] = float(probs[np.where(cand==pick)[0][0]])
            pval = pick
        cand_pb, probs_pb = build_weights_for_position("pb", post, bands["pb"], rec_mult_white, rec_mult_pb, used_white_vals, 1.0, True, None, None, gm_pb)
        if cand_pb.size == 0: return {}, {}
        pick_pb = int(cand_pb[int(np.argmax(probs_pb))]) if use_greedy else int(rng.choice(cand_pb, p=probs_pb))
        chosen["pb"] = pick_pb
        confs["pb"] = float(probs_pb[np.where(cand_pb==pick_pb)[0][0]])
        return chosen, confs
    ch, cf = pick_one(True, None)
    if ch:
        t = Ticket(ch["pos1"], ch["pos2"], ch["pos3"], ch["pos4"], ch["pos5"], ch["pb"], cf, overall_score(cf, OVERALL_SCORE_STYLE)); results.append(t); used_white_vals.update([t.pos1,t.pos2,t.pos3,t.pos4,t.pos5])
    while len(results) < n_sets:
        ch, cf = pick_one(False, None)
        if not ch: break
        t = Ticket(ch["pos1"], ch["pos2"], ch["pos3"], ch["pos4"], ch["pos5"], ch["pb"], cf, overall_score(cf, OVERALL_SCORE_STYLE)); results.append(t); used_white_vals.update([t.pos1,t.pos2,t.pos3,t.pos4,t.pos5])
    return results

def generate_tickets(df, bands, n_sets, lookback_years, prior_white, prior_pb, rec_half_life, rec_str_white, rec_str_pb, repeat_penalty_white, no_repeat_penalty, seed, mode):
    rng = np.random.default_rng(seed)
    end_date = df["date"].max()
    df_hist = df[(df["date"] <= end_date) & (df["date"] >= end_date - pd.DateOffset(years=lookback_years))].copy()
    post = build_posterior_counts(df_hist, prior_white, prior_pb)
    rec_mult_white, rec_mult_pb = compute_exponential_recency(df_hist, rec_half_life, rec_str_white, rec_str_pb)
    results = []; used_white_vals: Set[int] = set()
    def pick_one(use_greedy, prev_val):
        chosen = {}; confs = {}; pval = prev_val
        for key in ["pos1","pos2","pos3","pos4","pos5"]:
            cand, probs = build_weights_for_position(key, post, bands[key], rec_mult_white, rec_mult_pb, used_white_vals, repeat_penalty_white, no_repeat_penalty, pval)
            if cand.size == 0: return {}, {}
            pick = int(cand[int(np.argmax(probs))]) if use_greedy else int(rng.choice(cand, p=probs))
            chosen[key]=pick; confs[key]=float(probs[np.where(cand==pick)[0][0]]); pval=pick
        cand_pb, probs_pb = build_weights_for_position("pb", post, bands["pb"], rec_mult_white, rec_mult_pb, used_white_vals, 1.0, True, None)
        if cand_pb.size == 0: return {}, {}
        pick_pb = int(cand_pb[int(np.argmax(probs_pb))]) if use_greedy else int(rng.choice(cand_pb, p=probs_pb))
        chosen["pb"]=pick_pb; confs["pb"]=float(probs_pb[np.where(cand_pb==pick_pb)[0][0]]); return chosen, confs
    if mode=="greedy":
        for _ in range(n_sets):
            ch, cf = pick_one(True,None); if not ch: break
            t = Ticket(ch["pos1"],ch["pos2"],ch["pos3"],ch["pos4"],ch["pos5"],ch["pb"],cf,overall_score(cf, OVERALL_SCORE_STYLE)); results.append(t); used_white_vals.update([t.pos1,t.pos2,t.pos3,t.pos4,t.pos5])
        return results
    if mode=="hybrid":
        ch, cf = pick_one(True,None)
        if ch:
            t=Ticket(ch["pos1"],ch["pos2"],ch["pos3"],ch["pos4"],ch["pos5"],ch["pb"],cf,overall_score(cf, OVERALL_SCORE_STYLE)); results.append(t); used_white_vals.update([t.pos1,t.pos2,t.pos3,t.pos4,t.pos5])
        while len(results)<n_sets:
            ch, cf = pick_one(False,None); if not ch: break
            t=Ticket(ch["pos1"],ch["pos2"],ch["pos3"],ch["pos4"],ch["pos5"],ch["pb"],cf,overall_score(cf, OVERALL_SCORE_STYLE)); results.append(t); used_white_vals.update([t.pos1,t.pos2,t.pos3,t.pos4,t.pos5])
        return results
    while len(results)<n_sets:
        ch, cf = pick_one(False,None); if not ch: break
        t=Ticket(ch["pos1"],ch["pos2"],ch["pos3"],ch["pos4"],ch["pos5"],ch["pb"],cf,overall_score(cf, OVERALL_SCORE_STYLE)); results.append(t); used_white_vals.update([t.pos1,t.pos2,t.pos3,t.pos4,t.pos5])
    return results

def generate_all_mode(df, bands, n_sets, lookback_years, prior_white, prior_pb, rec_half_life, rec_str_white, rec_str_pb, repeat_penalty_white, no_repeat_penalty, seed, beam, overlap_penalty, top_pb_n, guide_pool, guide_strength_white, guide_strength_pb):
    if n_sets < 5: n_sets = 5
    slate = []
    slate += generate_tickets(df,bands,1,lookback_years,prior_white,prior_pb,rec_half_life,rec_str_white,rec_str_pb,repeat_penalty_white,no_repeat_penalty,seed,"greedy")
    slate += deterministic_generate(df,bands,1,lookback_years,prior_white,prior_pb,rec_half_life,rec_str_white,rec_str_pb,beam,overlap_penalty,top_pb_n)
    slate += generate_tickets_hybrid_det(df,bands,1,lookback_years,prior_white,prior_pb,rec_half_life,rec_str_white,rec_str_pb,repeat_penalty_white,no_repeat_penalty,seed,guide_pool,beam,guide_strength_white,guide_strength_pb)
    k = min(2, n_sets - len(slate)); 
    if k>0: slate += generate_tickets(df,bands,k,lookback_years,prior_white,prior_pb,rec_half_life,rec_str_white,rec_str_pb,repeat_penalty_white,no_repeat_penalty,seed,"hybrid")
    while len(slate) < n_sets:
        slate += generate_tickets(df,bands,1,lookback_years,prior_white,prior_pb,rec_half_life,rec_str_white,rec_str_pb,repeat_penalty_white,no_repeat_penalty,seed,"hybrid")
    return slate[:n_sets]

def parse_args():
    p = argparse.ArgumentParser(description="Powerball weighted picker v6")
    p.add_argument("--csv", required=True)
    p.add_argument("--sets", type=int, default=5, choices=[1,3,5,10])
    p.add_argument("--mode", choices=["sample","greedy","hybrid","deterministic","hybrid_det","all"], default="hybrid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lookback-years", type=int, default=LOOKBACK_YEARS)
    p.add_argument("--prior-strength-white", type=float, default=PRIOR_STRENGTH_WHITE)
    p.add_argument("--prior-strength-pb", type=float, default=PRIOR_STRENGTH_PB)
    p.add_argument("--recency-half-life", type=float, default=RECENCY_HALF_LIFE_DRAWS)
    p.add_argument("--recency-strength-white", type=float, default=RECENCY_STRENGTH_WHITE)
    p.add_argument("--recency-strength-pb", type=float, default=RECENCY_STRENGTH_PB)
    p.add_argument("--repeat-penalty", type=float, default=REPEAT_PENALTY_WHITE)
    p.add_argument("--no-repeat-penalty", action="store_true")
    p.add_argument("--beam", type=int, default=64)
    p.add_argument("--overlap-penalty", type=float, default=0.15)
    p.add_argument("--top-pb", type=int, default=1, choices=[1,2,3])
    p.add_argument("--guide-pool", type=int, default=20)
    p.add_argument("--guide-strength-white", type=float, default=0.5)
    p.add_argument("--guide-strength-pb", type=float, default=0.5)
    return p.parse_args()

def main():
    args = parse_args()
    df_all = load_draws(args.csv)
    bands = DEFAULT_BANDS.copy()
    if args.mode == "deterministic":
        tickets = deterministic_generate(df_all, bands, args.sets, args.lookback_years, args.prior_strength_white, args.prior_strength_pb, args.recency_half_life, args.recency_strength_white, args.recency_strength_pb, args.beam, args.overlap_penalty, args.top_pb)
    elif args.mode == "hybrid_det":
        tickets = generate_tickets_hybrid_det(df_all, bands, args.sets, args.lookback_years, args.prior_strength_white, args.prior_strength_pb, args.recency_half_life, args.recency_strength_white, args.recency_strength_pb, args.repeat_penalty, args.no_repeat_penalty, args.seed, args.guide_pool, args.beam, args.guide_strength_white, args.guide_strength_pb)
    elif args.mode == "all":
        n = max(5, args.sets)
        tickets = generate_all_mode(df_all, bands, n, args.lookback_years, args.prior_strength_white, args.prior_strength_pb, args.recency_half_life, args.recency_strength_white, args.recency_strength_pb, args.repeat_penalty, args.no_repeat_penalty, args.seed, args.beam, args.overlap_penalty, args.top_pb, args.guide_pool, args.guide_strength_white, args.guide_strength_pb)
    else:
        tickets = generate_tickets(df_all, bands, args.sets, args.lookback_years, args.prior_strength_white, args.prior_strength_pb, args.recency_half_life, args.recency_strength_white, args.recency_strength_pb, args.repeat_penalty, args.no_repeat_penalty, args.seed, args.mode)
    print("Generated tickets:\n")
    for i, t in enumerate(tickets, 1):
        whites = f"{t.pos1}, {t.pos2}, {t.pos3}, {t.pos4}, {t.pos5}"
        cp = t.conf_pos
        conf_str = (f"pos1 {cp['pos1']*100:.1f}%, pos2 {cp['pos2']*100:.1f}%, pos3 {cp['pos3']*100:.1f}%, pos4 {cp['pos4']*100:.1f}%, pos5 {cp['pos5']*100:.1f}%, PB {cp['pb']*100:.1f}%")
        print(f"{i}) {whites}  PB {t.pb}  |  Overall: {t.conf_overall*100:.2f}%  |  {conf_str}")

if __name__ == "__main__":
    main()
