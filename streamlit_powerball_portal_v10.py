#!/usr/bin/env python3
# Streamlit portal v10
# - Presets (Diversified, Concentrated, Balanced, PB Spread, Aggressive Deterministic, Gentle Hybrid)
# - Unique PB toggle (enforce different Powerballs across the slate)
# - Preset applicator updates sidebar controls in-place
# - Keeps overlap relax, proposals-per-attempt, logging, charts
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass
import math, datetime, os

WHITE_MAX = 69
PB_MAX = 26

DEFAULT_BANDS = {"pos1":(4,17),"pos2":(12,33),"pos3":(21,49),"pos4":(33,60),"pos5":(51,66),"pb":(4,22)}
LOOKBACK_YEARS=6
RECENCY_HALF_LIFE_DRAWS=45.0
RECENCY_STRENGTH_WHITE=0.35
RECENCY_STRENGTH_PB=0.35
PRIOR_STRENGTH_WHITE=120.0
PRIOR_STRENGTH_PB=26.0
REPEAT_PENALTY_WHITE=0.65

PRESETS = {
    "Balanced": {
        "overlap_limit": 1, "reroll_attempts": 120, "proposals_per_attempt": 3,
        "beam": 64, "guide_pool": 40, "gsw": 0.40, "gsp": 0.35, "pb_top_n": 2,
        "unique_pb": False
    },
    "Diversified": {
        "overlap_limit": 0, "reroll_attempts": 200, "proposals_per_attempt": 4,
        "beam": 64, "guide_pool": 50, "gsw": 0.35, "gsp": 0.30, "pb_top_n": 3,
        "unique_pb": True
    },
    "Concentrated": {
        "overlap_limit": 2, "reroll_attempts": 60, "proposals_per_attempt": 2,
        "beam": 128, "guide_pool": 50, "gsw": 0.60, "gsp": 0.50, "pb_top_n": 1,
        "unique_pb": False
    },
    "PB Spread": {
        "overlap_limit": 1, "reroll_attempts": 150, "proposals_per_attempt": 3,
        "beam": 64, "guide_pool": 40, "gsw": 0.40, "gsp": 0.30, "pb_top_n": 3,
        "unique_pb": True
    },
    "Aggressive Deterministic": {
        "overlap_limit": 1, "reroll_attempts": 100, "proposals_per_attempt": 3,
        "beam": 192, "guide_pool": 60, "gsw": 0.55, "gsp": 0.45, "pb_top_n": 2,
        "unique_pb": False
    },
    "Gentle Hybrid": {
        "overlap_limit": 1, "reroll_attempts": 120, "proposals_per_attempt": 3,
        "beam": 48, "guide_pool": 30, "gsw": 0.35, "gsp": 0.35, "pb_top_n": 2,
        "unique_pb": False
    },
}

def load_draws(file_like):
    df = pd.read_csv(file_like)
    req=["date","pos1","pos2","pos3","pos4","pos5","pb"]; missing=[c for c in req if c not in df.columns]
    if missing: raise ValueError(f"CSV missing columns: {missing}")
    df["date"]=pd.to_datetime(df["date"], errors="raise")
    for c in ["pos1","pos2","pos3","pos4","pos5","pb"]: df[c]=df[c].astype(int)
    return df.sort_values("date").reset_index(drop=True)

def frequencies(series: pd.Series, max_n: int) -> pd.Series:
    return series.value_counts().reindex(range(1, max_n+1), fill_value=0).sort_index()

def beta_pdf(a,b,x):
    if x<=0 or x>=1: return 0.0
    logB=math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    return math.exp((a-1)*math.log(x)+(b-1)*math.log(1-x)-logB)

def order_statistic_prior_white(pos_key, prior_strength):
    k=int(pos_key[-1]); a=float(k); b=float(6-k)
    xs=(np.arange(1,WHITE_MAX+1)-0.5)/WHITE_MAX
    pdf=np.array([beta_pdf(a,b,float(x)) for x in xs]); pdf=np.clip(pdf,1e-12,None)
    return (pdf/pdf.sum()*prior_strength).astype(float)

def prior_pb_uniform(prior_strength): return np.ones(PB_MAX,dtype=float)*(prior_strength/PB_MAX)
def normalize(a): s=float(a.sum()); return a/s if s>0 else np.zeros_like(a,dtype=float)

def build_posterior_counts(df_hist, prior_white_strength, prior_pb_strength):
    post={}
    for i in range(1,6):
        pos=f"pos{i}"
        counts=df_hist[pos].value_counts().reindex(range(1,WHITE_MAX+1), fill_value=0).sort_index().values.astype(float)
        post[pos]=counts+order_statistic_prior_white(pos, prior_white_strength)
    counts_pb=df_hist["pb"].value_counts().reindex(range(1,PB_MAX+1), fill_value=0).sort_index().values.astype(float)
    post["pb"]=counts_pb+prior_pb_uniform(prior_pb_strength); return post

def compute_exponential_recency(df_hist, half_life_draws, strength_white, strength_pb):
    if df_hist.empty: return np.ones(WHITE_MAX,dtype=float), np.ones(PB_MAX,dtype=float)
    n=len(df_hist); ages=(n-1)-np.arange(n); decay=np.power(0.5, ages/float(half_life_draws))
    rec_white=np.zeros(WHITE_MAX,dtype=float)
    for i,row in enumerate(df_hist.itertuples(index=False)):
        w=float(decay[i]); rec_white[row.pos1-1]+=w; rec_white[row.pos2-1]+=w; rec_white[row.pos3-1]+=w; rec_white[row.pos4-1]+=w; rec_white[row.pos5-1]+=w
    rec_pb=np.zeros(PB_MAX,dtype=float)
    for i,row in enumerate(df_hist.itertuples(index=False)): rec_pb[row.pb-1]+=float(decay[i])
    return 1.0+strength_white*rec_white, 1.0+strength_pb*rec_pb

def build_weights_for_position(pos_key, posterior_counts, band, recency_mult_white, recency_mult_pb, used_white_vals, repeat_penalty, no_repeat_penalty, prev_val=None, guide_mult_white=None, guide_mult_pb=None):
    is_pb=(pos_key=="pb"); lo,hi=band; start=lo if prev_val is None or is_pb else max(lo,prev_val+1)
    if start>hi: return np.array([],dtype=int), np.array([],dtype=float)
    candidates=np.arange(start,hi+1,dtype=int)
    base=posterior_counts[pos_key].copy(); base=base*(recency_mult_pb if is_pb else recency_mult_white)
    if not is_pb and guide_mult_white is not None and pos_key in guide_mult_white:
        gm=guide_mult_white[pos_key]
        if gm.shape[0]==base.shape[0]: base=base*gm
    if is_pb and guide_mult_pb is not None and guide_mult_pb.shape[0]==base.shape[0]: base=base*guide_mult_pb
    weights=base[candidates-1].astype(float)
    if not is_pb and not no_repeat_penalty and repeat_penalty!=1.0:
        repeat_mask=np.isin(candidates, list(used_white_vals)); weights[repeat_mask]*=repeat_penalty
    if weights.sum()<=0: weights=np.ones_like(weights,dtype=float)
    return candidates, normalize(weights)

@dataclass
class Ticket:
    pos1:int; pos2:int; pos3:int; pos4:int; pos5:int; pb:int
    conf_pos: dict; conf_overall: float
def overall_score(conf_pos): 
    keys=["pos1","pos2","pos3","pos4","pos5","pb"]; v=np.array([conf_pos[k] for k in keys],float); v=np.where(v>0,v,1e-12); return float(np.exp(np.mean(np.log(v))))

@dataclass
class BeamItem: whites: tuple; logp: float

def build_probs_for_position(pos_key, post, band, rec_mult_white, rec_mult_pb, prev_val=None):
    is_pb=(pos_key=="pb"); lo,hi=band; start=lo if prev_val is None or is_pb else max(lo,prev_val+1)
    if start>hi: return np.array([],dtype=int), np.array([],dtype=float)
    cand=np.arange(start,hi+1,dtype=int)
    base=post[pos_key].copy(); base=base*(rec_mult_pb if is_pb else rec_mult_white)
    p=base[cand-1].astype(float); s=p.sum(); return cand, (p/s if s>0 else p)

def top_k_white_sequences(post,bands,rec_mult_white,K,beam):
    cand1,p1=build_probs_for_position("pos1",post,bands["pos1"],rec_mult_white,np.ones(PB_MAX),None)
    beams=[BeamItem((int(cand1[i]),0,0,0,0), float(np.log(max(p1[i],1e-18)))) for i in range(len(cand1))]; beams.sort(key=lambda x:x.logp, reverse=True); beams=beams[:beam]
    def extend(beams_in,pos_key,idx):
        out=[]
        for b in beams_in:
            prev=b.whites[idx-2] if idx>1 else None
            cand,p=build_probs_for_position(pos_key,post,bands[pos_key],rec_mult_white,np.ones(PB_MAX),prev)
            for i in range(len(cand)):
                val=int(cand[i])
                if prev is not None and val<=prev: continue
                logp=b.logp+float(np.log(max(p[i],1e-18)))
                if idx==2: whites=(b.whites[0],val,0,0,0)
                elif idx==3: whites=(b.whites[0],b.whites[1],val,0,0)
                elif idx==4: whites=(b.whites[0],b.whites[1],b.whites[2],val,0)
                else: whites=(b.whites[0],b.whites[1],b.whites[2],b.whites[3],val)
                out.append(BeamItem(whites,logp))
        out.sort(key=lambda x:x.logp, reverse=True); return out[:beam]
    beams=extend(beams,"pos2",2); beams=extend(beams,"pos3",3); beams=extend(beams,"pos4",4); beams=extend(beams,"pos5",5); beams.sort(key=lambda x:x.logp, reverse=True); return beams[:max(1,K*5)]

def top_pb_choices(post, band_pb, rec_mult_pb, top_n):
    cand,p=build_probs_for_position("pb",post,band_pb,np.ones(WHITE_MAX),rec_mult_pb,None)
    items=[(int(cand[i]), float(p[i])) for i in range(len(cand))]; items.sort(key=lambda x:x[1], reverse=True); return items[:max(1,top_n)]

def compute_guidance_multipliers(post,bands,rec_mult_white,rec_mult_pb,guide_pool,beam,gsw,gsp):
    whites_pool=top_k_white_sequences(post,bands,rec_mult_white,guide_pool,beam)
    if not whites_pool:
        gm_white={f"pos{i}":np.ones(WHITE_MAX,float) for i in range(1,6)}; gm_pb=np.ones(PB_MAX,float); return gm_white,gm_pb
    lp=np.array([w.logp for w in whites_pool],float); lp-=lp.max(); wts=np.exp(lp); wts/=wts.sum()
    gm_white={f"pos{i}":np.zeros(WHITE_MAX,float) for i in range(1,6)}
    for w,wt in zip(whites_pool,wts):
        vals=list(w.whites)
        for i,val in enumerate(vals, start=1):
            if 1<=val<=WHITE_MAX: gm_white[f"pos{i}"][val-1]+=wt
    for k in gm_white.keys():
        prob=gm_white[k]
        if prob.sum()<=0: gm_white[k]=np.ones(WHITE_MAX,float); continue
        prob/=prob.sum(); mean=1.0/WHITE_MAX; ratio=np.where(prob>0, prob/mean, 0.0); gm_white[k]=(1.0-gsw)+gsw*ratio
    cand_pb,p_pb=build_probs_for_position("pb",post,bands["pb"],np.ones(WHITE_MAX),rec_mult_pb,None)
    gm_pb=np.ones(PB_MAX,float)
    if cand_pb.size>0 and p_pb.sum()>0:
        probs=p_pb/p_pb.sum(); mean=1.0/PB_MAX; ratio=probs/mean; gm_pb[cand_pb-1]=(1.0-gsp)+gsp*ratio
    return gm_white, gm_pb

def generate_tickets(df,bands,n_sets,lookback_years,prior_white,prior_pb,half_life,rec_str_white,rec_str_pb,repeat_pen,no_repeat_penalty,seed,mode):
    rng=np.random.default_rng(seed); end=df["date"].max()
    df_hist=df[(df["date"]<=end)&(df["date"]>=end-pd.DateOffset(years=lookback_years))].copy()
    post=build_posterior_counts(df_hist,prior_white,prior_pb); recw,recp=compute_exponential_recency(df_hist,half_life,rec_str_white,rec_str_pb)
    results=[]; used=set()
    def pick_one(use_greedy,prev):
        chosen={}; confs={}; pval=prev
        for key in ["pos1","pos2","pos3","pos4","pos5"]:
            cand,probs=build_weights_for_position(key,post,bands[key],recw,recp,used,repeat_pen,no_repeat_penalty,pval)
            if cand.size==0:
                return {},{}
            if use_greedy:
                pick=int(cand[int(np.argmax(probs))])
            else:
                pick=int(rng.choice(cand,p=probs))
            chosen[key]=pick; confs[key]=float(probs[np.where(cand==pick)[0][0]]); pval=pick
        cand_pb,ppb=build_weights_for_position("pb",post,bands["pb"],recw,recp,used,1.0,True,None)
        if cand_pb.size==0:
            return {},{}
        if use_greedy:
            pick_pb=int(cand_pb[int(np.argmax(ppb))])
        else:
            pick_pb=int(rng.choice(cand_pb,p=ppb))
        chosen["pb"]=pick_pb; confs["pb"]=float(ppb[np.where(cand_pb==pick_pb)[0][0]]); return chosen,confs
    if mode=="greedy":
        for _ in range(n_sets):
            ch,cf=pick_one(True,None)
            if not ch:
                break
            t=Ticket(ch["pos1"],ch["pos2"],ch["pos3"],ch["pos4"],ch["pos5"],ch["pb"],cf,overall_score(cf)); results.append(t); used.update([t.pos1,t.pos2,t.pos3,t.pos4,t.pos5])
        return results
    if mode=="hybrid":
        ch,cf=pick_one(True,None)
        if ch:
            t=Ticket(ch["pos1"],ch["pos2"],ch["pos3"],ch["pos4"],ch["pos5"],ch["pb"],cf,overall_score(cf)); results.append(t); used.update([t.pos1,t.pos2,t.pos3,t.pos4,t.pos5])
        while len(results)<n_sets:
            ch,cf=pick_one(False,None)
            if not ch:
                break
            t=Ticket(ch["pos1"],ch["pos2"],ch["pos3"],ch["pos4"],ch["pos5"],ch["pb"],cf,overall_score(cf)); results.append(t); used.update([t.pos1,t.pos2,t.pos3,t.pos4,t.pos5])
        return results
    while len(results)<n_sets:
        ch,cf=pick_one(False,None)
        if not ch:
            break
        t=Ticket(ch["pos1"],ch["pos2"],ch["pos3"],ch["pos4"],ch["pos5"],ch["pb"],cf,overall_score(cf)); results.append(t); used.update([t.pos1,t.pos2,t.pos3,t.pos4,t.pos5])
    return results

def deterministic_generate(df,bands,n_sets,lookback_years,prior_white,prior_pb,half_life,rec_str_white,rec_str_pb,beam,overlap_penalty,top_pb_n):
    end=df["date"].max(); df_hist=df[(df["date"]<=end)&(df["date"]>=end-pd.DateOffset(years=lookback_years))].copy()
    post=build_posterior_counts(df_hist,prior_white,prior_pb); recw,recp=compute_exponential_recency(df_hist,half_life,rec_str_white,rec_str_pb)
    whites_pool=top_k_white_sequences(post,bands,recw,n_sets,beam); pb_top=top_pb_choices(post,bands["pb"],recp,top_pb_n)
    cands=[]
    for w in whites_pool:
        confs={}; prev=None
        for idx,key in enumerate(["pos1","pos2","pos3","pos4","pos5"]):
            cand,p=build_probs_for_position(key,post,bands[key],recw,recp,prev); chosen=w.whites[idx]; j=np.where(cand==chosen)[0]; confs[key]=float(p[j[0]]) if j.size>0 else 0.0; prev=chosen
        for pb_val,_ in pb_top:
            cand_pb,pb=build_probs_for_position("pb",post,bands["pb"],recw,recp,None); j=np.where(cand_pb==pb_val)[0]; confs2=dict(confs); confs2["pb"]=float(pb[j[0]]) if j.size>0 else 0.0
            t=Ticket(w.whites[0],w.whites[1],w.whites[2],w.whites[3],w.whites[4],pb_val,confs2,overall_score(confs2)); t.base_logp=float(w.logp+np.log(max(confs2["pb"],1e-18))); cands.append(t)
    cands.sort(key=lambda t:(t.base_logp,t.pos1,t.pos2,t.pos3,t.pos4,t.pos5,t.pb), reverse=True)
    selected=[]; used=[]
    while len(selected)<n_sets and cands:
        best=None; best_score=-1e18
        for t in cands:
            overlap=0
            for s in used: overlap+=len(s & set([t.pos1,t.pos2,t.pos3,t.pos4,t.pos5]))
            score=t.base_logp-0.15*float(overlap)
            if score>best_score: best=t; best_score=score
        if best is None: break
        selected.append(best); used.append(set([best.pos1,best.pos2,best.pos3,best.pos4,best.pos5]))
        cands=[c for c in cands if (c.pos1,c.pos2,c.pos3,c.pos4,c.pos5,c.pb)!=(best.pos1,best.pos2,best.pos3,best.pos4,best.pos5,best.pb)]
    return selected

def generate_tickets_hybrid_det(df,bands,n_sets,lookback_years,prior_white,prior_pb,half_life,rec_str_white,rec_str_pb,repeat_pen,no_repeat_penalty,seed,guide_pool,beam,gsw,gsp):
    rng=np.random.default_rng(seed); end=df["date"].max()
    df_hist=df[(df["date"]<=end)&(df["date"]>=end-pd.DateOffset(years=lookback_years))].copy()
    post=build_posterior_counts(df_hist,prior_white,prior_pb); recw,recp=compute_exponential_recency(df_hist,half_life,rec_str_white,rec_str_pb)
    gm_white,gm_pb=compute_guidance_multipliers(post,bands,recw,recp,guide_pool,beam,gsw,gsp)
    results=[]; used=set()
    def pick_one(use_greedy,prev):
        chosen={}; confs={}; pval=prev
        for key in ["pos1","pos2","pos3","pos4","pos5"]:
            cand,probs=build_weights_for_position(key,post,bands[key],recw,recp,used,repeat_pen,no_repeat_penalty,pval,gm_white,None)
            if cand.size==0:
                return {},{}
            if use_greedy:
                pick=int(cand[int(np.argmax(probs))])
            else:
                pick=int(rng.choice(cand,p=probs))
            chosen[key]=pick; confs[key]=float(probs[np.where(cand==pick)[0][0]]); pval=pick
        cand_pb,ppb=build_weights_for_position("pb",post,bands["pb"],recw,recp,used,1.0,True,None,None,gm_pb)
        if cand_pb.size==0:
            return {},{}
        if use_greedy:
            pick_pb=int(cand_pb[int(np.argmax(ppb))])
        else:
            pick_pb=int(rng.choice(cand_pb,p=ppb))
        chosen["pb"]=pick_pb; confs["pb"]=float(ppb[np.where(cand_pb==pick_pb)[0][0]]); return chosen,confs
    ch,cf=pick_one(True,None)
    if ch:
        t=Ticket(ch["pos1"],ch["pos2"],ch["pos3"],ch["pos4"],ch["pos5"],ch["pb"],cf,overall_score(cf)); results.append(t); used.update([t.pos1,t.pos2,t.pos3,t.pos4,t.pos5])
    while len(results)<n_sets:
        ch,cf=pick_one(False,None)
        if not ch:
            break
        t=Ticket(ch["pos1"],ch["pos2"],ch["pos3"],ch["pos4"],ch["pos5"],ch["pb"],cf,overall_score(cf)); results.append(t); used.update([t.pos1,t.pos2,t.pos3,t.pos4,t.pos5])
    return results

# ---- Slate helpers ----
def white_overlap(a: Ticket, b: Ticket) -> int:
    return len({a.pos1,a.pos2,a.pos3,a.pos4,a.pos5} & {b.pos1,b.pos2,b.pos3,b.pos4,b.pos5})

def enforce_constraints(initial: List[Ticket], want_n: int, limit: int, unique_pb: bool, make_more_fn, max_attempts: int, seed: int, proposals_per_attempt:int=3, relax_if_needed: bool=True) -> List[Ticket]:
    selected: List[Ticket] = []

    def ok_with_selected(t: Ticket, selected: List[Ticket], limit: int, unique_pb: bool) -> bool:
        if unique_pb and any(t.pb == s.pb for s in selected): return False
        return all(white_overlap(t, s) <= limit for s in selected)

    # Seed from initial
    for t in initial:
        if ok_with_selected(t, selected, limit, unique_pb):
            selected.append(t)
        if len(selected) >= want_n:
            break

    attempts=0; reroll=0; cur_limit=limit
    def try_fill(cur_limit, attempts_left, start_seed):
        nonlocal attempts, reroll, selected
        while len(selected)<want_n and attempts<attempts_left:
            new = make_more_fn(start_seed + reroll)
            reroll += 1; attempts += 1
            for t in new[:proposals_per_attempt]:
                if ok_with_selected(t, selected, cur_limit, unique_pb):
                    selected.append(t); break

    try_fill(max_attempts, max_attempts, seed)
    while relax_if_needed and len(selected)<want_n and cur_limit<4:
        cur_limit += 1
        try_fill(cur_limit, max_attempts, seed+reroll)

    return selected

# ---- UI ----
st.set_page_config(page_title="Powerball Portal v10", layout="wide")
st.title("Powerball Weighted Picks Portal (v10)")

with st.sidebar:
    # --- Presets FIRST ---
    st.subheader("Presets")
    preset = st.selectbox("Choose preset", list(PRESETS.keys()), index=0, key="preset_k")
    if st.button("Apply preset"):
        cfg = PRESETS[st.session_state["preset_k"]]
        st.session_state["overlap_k"] = int(cfg["overlap_limit"])
        st.session_state["reroll_k"] = int(cfg["reroll_attempts"])
        st.session_state["proposals_k"] = int(cfg["proposals_per_attempt"])
        st.session_state["beam_k"] = int(cfg["beam"])
        st.session_state["gpool_k"] = int(cfg["guide_pool"])
        st.session_state["gsw_k"] = float(cfg["gsw"])
        st.session_state["gsp_k"] = float(cfg["gsp"])
        st.session_state["pbtop_k"] = int(cfg["pb_top_n"])
        st.session_state["unique_pb_k"] = bool(cfg["unique_pb"])
        st.rerun()

    # --- Then the rest of your controls ---
    st.header("Controls")
    st.selectbox("Number of sets", options=[5,6,7,8,9,10], index=0, key="sets_k")
    st.selectbox("Mode", options=["ALL","deterministic","hybrid_det","hybrid","sample","greedy"], index=0, key="mode_k")
    st.number_input("Random seed (ignored for deterministic)", value=42, step=1, key="seed_k")

    st.subheader("Windows and priors")
    st.number_input("Lookback years", min_value=1, max_value=9, value=int(LOOKBACK_YEARS), step=1, key="lookback_k")
    st.number_input("Prior strength white", min_value=0.0, value=float(PRIOR_STRENGTH_WHITE), step=10.0, key="priorw_k")
    st.number_input("Prior strength PB", min_value=0.0, value=float(PRIOR_STRENGTH_PB), step=2.0, key="priorpb_k")

    st.subheader("Recency")
    st.number_input("Half-life in draws", min_value=5.0, max_value=200.0, value=float(RECENCY_HALF_LIFE_DRAWS), step=1.0, key="halflife_k")
    st.number_input("Recency strength white", min_value=0.0, max_value=2.0, value=float(RECENCY_STRENGTH_WHITE), step=0.05, format="%.2f", key="recw_k")
    st.number_input("Recency strength PB", min_value=0.0, max_value=2.0, value=float(RECENCY_STRENGTH_PB), step=0.05, format="%.2f", key="recpb_k")

    st.subheader("Diversification")
    st.number_input("Repeat penalty (white)", min_value=0.1, max_value=1.0, value=float(REPEAT_PENALTY_WHITE), step=0.05, key="repeat_k")
    st.checkbox("Disable repeat penalty", value=False, key="norepeat_k")
    st.slider("Max shared white balls across tickets", min_value=0, max_value=4, value=1, step=1, key="overlap_k")
    st.number_input("Max reroll attempts", min_value=0, max_value=400, value=120, step=10, key="reroll_k")
    st.number_input("Proposals per attempt", min_value=1, max_value=5, value=3, step=1, key="proposals_k")
    st.checkbox("Unique PB across slate", value=False, key="unique_pb_k")

    st.subheader("Deterministic + Guidance")
    st.number_input("Beam width", min_value=8, max_value=512, value=64, step=8, key="beam_k")
    st.number_input("Guidance pool (top sequences)", min_value=10, max_value=200, value=40, step=5, key="gpool_k")
    st.number_input("Guide strength white (0..1)", min_value=0.0, max_value=1.0, value=0.4, step=0.05, key="gsw_k")
    st.number_input("Guide strength PB (0..1)", min_value=0.0, max_value=1.0, value=0.35, step=0.05, key="gsp_k")
    st.number_input("PB top-N for deterministic", min_value=1, max_value=3, value=2, step=1, key="pbtop_k")

    st.subheader("Logging")
    st.checkbox("Also save log to disk (.md)", value=False, key="save_log_k")
    st.text_input("Log folder (created if missing)", value="logs", key="logdir_k")


st.write("Upload your CSV of drawings (date,pos1,pos2,pos3,pos4,pos5,pb).")
file = st.file_uploader("CSV file", type=["csv"], accept_multiple_files=False)
if file is None:
    st.info("Waiting for CSV upload."); st.stop()
try:
    df_all = load_draws(file)
except Exception as e:
    st.error(f"Failed to load CSV: {e}"); st.stop()

st.success(f"Loaded {len(df_all)} drawings. Range: {df_all['date'].min().date()} to {df_all['date'].max().date()}")

st.markdown("### Bands")
c1,c2,c3 = st.columns(3)
with c1:
    pos1_band = st.slider("Pos1", 1, WHITE_MAX, DEFAULT_BANDS["pos1"], key="b_pos1")
    pos2_band = st.slider("Pos2", 1, WHITE_MAX, DEFAULT_BANDS["pos2"], key="b_pos2")
with c2:
    pos3_band = st.slider("Pos3", 1, WHITE_MAX, DEFAULT_BANDS["pos3"], key="b_pos3")
    pos4_band = st.slider("Pos4", 1, WHITE_MAX, DEFAULT_BANDS["pos4"], key="b_pos4")
with c3:
    pos5_band = st.slider("Pos5", 1, WHITE_MAX, DEFAULT_BANDS["pos5"], key="b_pos5")
    pb_band = st.slider("PB", 1, PB_MAX, DEFAULT_BANDS["pb"], key="b_pb")
bands = {"pos1":st.session_state["b_pos1"],"pos2":st.session_state["b_pos2"],"pos3":st.session_state["b_pos3"],"pos4":st.session_state["b_pos4"],"pos5":st.session_state["b_pos5"],"pb":st.session_state["b_pb"]}

def plot_freq(series, max_n, title):
    x=np.arange(1,max_n+1); counts=frequencies(series,max_n).reindex(x,fill_value=0).values
    fig=plt.figure(figsize=(10,2.8)); plt.bar(x,counts); plt.xlabel("Number"); plt.ylabel("Times drawn"); plt.title(title); plt.tight_layout(); return fig

g1,g2 = st.columns(2)
with g1:
    st.pyplot(plot_freq(df_all["pos1"], WHITE_MAX, "POS1 since file start"))
    st.pyplot(plot_freq(df_all["pos2"], WHITE_MAX, "POS2 since file start"))
    st.pyplot(plot_freq(df_all["pos3"], WHITE_MAX, "POS3 since file start"))
with g2:
    st.pyplot(plot_freq(df_all["pos4"], WHITE_MAX, "POS4 since file start"))
    st.pyplot(plot_freq(df_all["pos5"], WHITE_MAX, "POS5 since file start"))
    st.pyplot(plot_freq(df_all["pb"], PB_MAX, "PB since file start"))

# Pull all sidebar values
sets            = int(st.session_state["sets_k"])
mode            = st.session_state["mode_k"]
seed            = int(st.session_state["seed_k"])
lookback_years  = int(st.session_state["lookback_k"])
prior_white     = float(st.session_state["priorw_k"])
prior_pb        = float(st.session_state["priorpb_k"])
half_life       = float(st.session_state["halflife_k"])
rec_str_white   = float(st.session_state["recw_k"])
rec_str_pb      = float(st.session_state["recpb_k"])
repeat_pen      = float(st.session_state["repeat_k"])
no_repeat_penalty = bool(st.session_state["norepeat_k"])
overlap_limit   = int(st.session_state["overlap_k"])
reroll_attempts = int(st.session_state["reroll_k"])
proposals_per_attempt = int(st.session_state["proposals_k"])
unique_pb       = bool(st.session_state["unique_pb_k"])
beam            = int(st.session_state["beam_k"])
guide_pool      = int(st.session_state["gpool_k"])
gsw             = float(st.session_state["gsw_k"])
gsp             = float(st.session_state["gsp_k"])
pb_top_n        = int(st.session_state["pbtop_k"])
save_log        = bool(st.session_state["save_log_k"])
log_dir         = st.session_state["logdir_k"]

def run_mode(m, n_sets, rng_seed):
    if m=="deterministic":
        return deterministic_generate(df_all, bands, n_sets, int(lookback_years), float(prior_white), float(prior_pb), float(half_life), float(rec_str_white), float(rec_str_pb), int(beam), 0.15, int(pb_top_n))
    if m=="hybrid_det":
        return generate_tickets_hybrid_det(df_all, bands, n_sets, int(lookback_years), float(prior_white), float(prior_pb), float(half_life), float(rec_str_white), float(rec_str_pb), float(repeat_pen), bool(no_repeat_penalty), int(rng_seed), int(guide_pool), int(beam), float(gsw), float(gsp))
    if m in ("hybrid","sample","greedy"):
        return generate_tickets(df_all, bands, n_sets, int(lookback_years), float(prior_white), float(prior_pb), float(half_life), float(rec_str_white), float(rec_str_pb), float(repeat_pen), bool(no_repeat_penalty), int(rng_seed), m)
    return []

def make_more_fn_factory(mode_name, per_try_sets):
    def _fn(rng_seed):
        return run_mode(mode_name, per_try_sets, rng_seed)
    return _fn

@dataclass
class TicketRow:
    ticket:int; pos1:int; pos2:int; pos3:int; pos4:int; pos5:int; pb:int; overall_conf:float; confs:dict

def slate_to_dataframe(tickets: List[Ticket]) -> pd.DataFrame:
    rows=[]
    for i,t in enumerate(tickets,1):
        rows.append({"ticket":i,"pos1":t.pos1,"pos2":t.pos2,"pos3":t.pos3,"pos4":t.pos4,"pos5":t.pos5,"pb":t.pb,
                     "overall_conf_%":round(100.0*t.conf_overall,2),
                     "pos1_conf_%":round(100.0*t.conf_pos["pos1"],2),
                     "pos2_conf_%":round(100.0*t.conf_pos["pos2"],2),
                     "pos3_conf_%":round(100.0*t.conf_pos["pos3"],2),
                     "pos4_conf_%":round(100.0*t.conf_pos["pos4"],2),
                     "pos5_conf_%":round(100.0*t.conf_pos["pos5"],2),
                     "pb_conf_%":round(100.0*t.conf_pos["pb"],2)})
    return pd.DataFrame(rows)

def band_freq_top(df_hist: pd.DataFrame, band: Tuple[int,int], col: str, top_k:int=5) -> List[Tuple[int,int]]:
    lo,hi=band; s=df_hist[col]; vc=s[s.between(lo,hi)].value_counts().sort_index()
    if vc.empty: return []
    vc=vc.sort_values(ascending=False).head(top_k)
    return list(zip(vc.index.tolist(), vc.values.tolist()))

def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"]*len(cols)) + " |")
    for _, row in df.iterrows():
        cells = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def slate_overlap_summary(tickets_df: pd.DataFrame) -> List[str]:
    lines = []
    n = len(tickets_df)
    if n <= 1:
        return ["(only one ticket)"]
    overlaps = []
    for i in range(n):
        for j in range(i+1, n):
            a = tickets_df.loc[i]; b = tickets_df.loc[j]
            ov = len({a['pos1'],a['pos2'],a['pos3'],a['pos4'],a['pos5']} & {b['pos1'],b['pos2'],b['pos3'],b['pos4'],b['pos5']})
            overlaps.append((i+1, j+1, ov))
    max_ov = max(o[2] for o in overlaps); min_ov = min(o[2] for o in overlaps); mean_ov = sum(o[2] for o in overlaps) / len(overlaps)
    lines.append(f"- Pairwise overlap (whites): min={min_ov}, max={max_ov}, avg={mean_ov:.2f}")
    heavy = [o for o in overlaps if o[2] >= 2]
    if heavy:
        lines.append("- Pairs with overlap ≥ 2: " + ", ".join([f"{a}-{b}({ov})" for a,b,ov in heavy]))
    pb_counts = tickets_df['pb'].value_counts().sort_index()
    lines.append("- PB spread: " + ", ".join([f"{k}×{v}" for k,v in pb_counts.items()]))
    unique_whites = set()
    for i in range(n):
        r = tickets_df.loc[i]; unique_whites.update([r['pos1'], r['pos2'], r['pos3'], r['pos4'], r['pos5']])
    lines.append(f"- Unique white balls in slate: {len(unique_whites)}")
    return lines

def build_log_markdown(df: pd.DataFrame, df_hist: pd.DataFrame, tickets_df: pd.DataFrame, params: dict, bands: dict) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"# Powerball Portal Run Log")
    lines.append("")
    lines.append(f"- Timestamp: {ts}")
    lines.append(f"- Draws loaded: {len(df)}  |  Range: {df['date'].min().date()} → {df['date'].max().date()}")
    lines.append("")
    lines.append("## Settings")
    for k,v in params.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Bands")
    for k,v in bands.items():
        lines.append(f"- {k}: {v[0]}–{v[1]}")
    lines.append("")
    lines.append("## Band Top Frequencies (lookback window)")
    for i,col in enumerate(["pos1","pos2","pos3","pos4","pos5"], start=1):
        top = band_freq_top(df_hist, bands[col], col, top_k=7)
        lines.append(f"- **{col.upper()}**: " + (", ".join([f"{n}({c})" for n,c in top]) if top else "none"))
    top_pb = band_freq_top(df_hist, bands["pb"], "pb", top_k=7)
    lines.append(f"- **PB**: " + (", ".join([f"{n}({c})" for n,c in top_pb]) if top_pb else "none"))
    lines.append("")
    lines.append("## Tickets")
    lines.append("")
    lines.append(df_to_markdown(tickets_df))
    lines.append("")
    lines.append("## Slate overlap summary")
    lines.extend(slate_overlap_summary(tickets_df))
    lines.append("")
    return "\n".join(lines)

if st.button("Generate tickets"):
    end=df_all["date"].max()
    df_hist=df_all[(df_all["date"]<=end)&(df_all["date"]>=end-pd.DateOffset(years=int(lookback_years)))].copy()

    if mode=="ALL":
        n=max(5,int(sets))
        greedy_1 = run_mode("greedy", 1, seed)
        det_1 = run_mode("deterministic", 1, seed)
        hdet_1 = run_mode("hybrid_det", 1, seed)
        hyb_2 = run_mode("hybrid", min(2, n-3), seed)
        initial = greedy_1 + det_1 + hdet_1 + hyb_2
        tickets = enforce_constraints(
            initial, n, int(overlap_limit), bool(unique_pb),
            make_more_fn_factory("hybrid", int(proposals_per_attempt)),
            int(reroll_attempts),
            int(seed),
            proposals_per_attempt=int(proposals_per_attempt),
            relax_if_needed=True
        )
    else:
        initial = run_mode(mode, int(sets), seed)
        tickets = enforce_constraints(
            initial, int(sets), int(overlap_limit), bool(unique_pb),
            make_more_fn_factory(mode if mode in ("hybrid","sample","greedy","hybrid_det") else "hybrid", int(proposals_per_attempt)),
            int(reroll_attempts),
            int(seed),
            proposals_per_attempt=int(proposals_per_attempt),
            relax_if_needed=True
        )

    if not tickets:
        st.warning("No tickets generated. Try relaxing overlap limit, turning off Unique PB, increasing reroll attempts, widening bands, or lowering guide strengths.")
    else:
        out_df = slate_to_dataframe(tickets)
        st.dataframe(out_df, use_container_width=True)

        st.download_button("Download tickets as CSV", data=out_df.to_csv(index=False).encode("utf-8"), file_name="generated_tickets.csv", mime="text/csv")

        params = {
            "mode": mode,
            "preset": st.session_state.get("preset_k","(custom)"),
            "sets": int(sets),
            "seed": int(seed),
            "lookback_years": int(lookback_years),
            "prior_white": float(prior_white),
            "prior_pb": float(prior_pb),
            "half_life": float(half_life),
            "rec_str_white": float(rec_str_white),
            "rec_str_pb": float(rec_str_pb),
            "repeat_penalty_white": float(repeat_pen),
            "no_repeat_penalty": bool(no_repeat_penalty),
            "overlap_limit": int(overlap_limit),
            "unique_pb": bool(unique_pb),
            "reroll_attempts": int(reroll_attempts),
            "proposals_per_attempt": int(proposals_per_attempt),
            "beam": int(beam),
            "guide_pool": int(guide_pool),
            "guide_strength_white": float(gsw),
            "guide_strength_pb": float(gsp),
            "pb_top_n_deterministic": int(pb_top_n)
        }
        log_md = build_log_markdown(df_all, df_hist, out_df, params, bands)
        default_name = f"pb_portal_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        st.download_button("Download detailed run log (.md)", data=log_md.encode("utf-8"), file_name=default_name, mime="text/markdown")
        with st.expander("Preview run log (Markdown)"):
            st.code(log_md, language="markdown")
        if save_log:
            try:
                os.makedirs(log_dir, exist_ok=True)
                fpath = os.path.join(log_dir, default_name)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(log_md)
                st.success(f"Saved log to: {fpath}")
            except Exception as e:
                st.error(f"Failed to save log: {e}")
