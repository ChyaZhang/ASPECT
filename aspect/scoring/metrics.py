from __future__ import annotations

import json
import math
import re

from .task_rules import (
    OBSERVE_RE, FINAL_RE, _strip_think, extract_quantities,
    step_score, consistency,
)

KAPPA = math.log(3.0)
COUNTING = {"argmax_region", "multi_hop", "region_compare", "type_compare",
            "compartment_spatial"}
LETTERS = "ABCDEFGHIJKL"


def _map_opt(cand, options):
    cand = re.split(r"[<\n]", cand)[0].strip().strip("*`_ 。.")
    if not cand:
        return None
    lm = re.match(r"^\(?([A-L])[\.\)]\s*(.+)$", cand, re.I)
    if lm and options:
        rest = lm.group(2).strip()
        if rest in options:
            return rest
        i = LETTERS.index(lm.group(1).upper())
        return options[i] if i < len(options) else None
    if options and cand in options:
        return cand
    m2 = re.match(r"^\(?([A-L])[\.\)\s]*$", cand, re.I)
    if m2 and options:
        i = LETTERS.index(m2.group(1).upper())
        return options[i] if i < len(options) else None
    if options:
        hit = [o for o in options if o.lower() == cand.lower()]
        if hit:
            return hit[0]
    return cand or None


def _extract_pred(text, options=None):
    tail = _strip_think(text if isinstance(text, str) else "")
    finals = FINAL_RE.findall(tail)
    if finals:
        return _map_opt(finals[-1], options)
    m = re.search(r"<answer>\s*(.*?)\s*(?:</answer>|$)", tail, re.I | re.S)
    if m:
        return _map_opt(m.group(1), options)
    lines = [l.strip() for l in tail.splitlines() if l.strip()]
    return _map_opt(lines[-1], options) if lines else None

def smooth_p(claimed, gold_q, kappa=KAPPA):
    if not gold_q:
        return 0.0, 0
    claimed = claimed or {}
    tot = 0.0
    for k, tv in gold_q.items():
        cv = float(claimed.get(k, 0.0))
        d = abs(cv - float(tv)) / max(1.0, 0.1 * abs(float(tv)))
        tot += 2.0 / (1.0 + math.exp(min(kappa * d, 700.0)))
    return tot / len(gold_q), len(gold_q)


def _dup_keys(json_text):
    flag = {"d": False}
    def hook(pairs):
        seen = set()
        for k, _ in pairs:
            if k in seen:
                flag["d"] = True
            seen.add(k)
        return dict(pairs)
    try:
        json.loads(json_text, object_pairs_hook=hook)
    except Exception:
        return False
    return flag["d"]


def struct_valid(text, options=None):
    t = _strip_think(text if isinstance(text, str) else "")
    obs = OBSERVE_RE.findall(t)
    claimed = extract_quantities(text)
    ans = _extract_pred(text, options)
    flags = {
        "parseable":  claimed is not None,
        "one_observe": len(obs) == 1,
        "nonneg":     claimed is not None and all(v >= 0 for v in claimed.values()),
        "no_dupkey":  not (bool(obs) and _dup_keys(obs[0])),
        "has_answer": ans is not None,
    }
    vs = (flags["parseable"] and flags["nonneg"] and flags["no_dupkey"]
          and flags["has_answer"])
    return bool(vs), flags


_CANON = [
    (r"normal epitheli",                    "epithelium_normal"),
    (r"tumou?r|malignan|neoplas|carcinoma", "tumor"),
    (r"non-?lymphocytic|inflammatory \(non", "inflammatory_other"),
    (r"epitheli",                           "epithelial_like"),
    (r"lympho",                             "lymphocyte"),
    (r"inflammat",                          "inflammatory"),
    (r"stroma|connective|spindle|fibrobl",  "stromal_like"),
]
_INFL_FAM = {"inflammatory": "inflammatory_other",
             "inflammatory_other": "inflammatory"}


def _canon_type(text):
    for pat, key in _CANON:
        if re.search(pat, text, re.I):
            return key
    return None


def _resolve_key(phrase, avail):
    c = _canon_type(phrase)
    if c is None:
        return None
    if c in avail:
        return c
    if c in _INFL_FAM and _INFL_FAM[c] in avail:
        return _INFL_FAM[c]
    return c


def _parse_bands(question):
    lo = hi = None
    m = re.search(r"few\s*=\s*\d+\s*[-–]\s*(\d+)", question, re.I)
    if m:
        lo = int(m.group(1))
    m = re.search(r"moderate\s*=\s*\d+\s*[-–]\s*(\d+)", question, re.I)
    if m:
        hi = int(m.group(1))
    return (lo if lo is not None else 10, hi if hi is not None else 50)


def _fofC_s1(claimed, skill, gold_q, options, question):
    claimed = claimed or {}
    q = question or ""
    avail = list(claimed) + list(gold_q or {})
    if skill == "what_dominant":
        cands = list(options) if options else list(gold_q or {})
        cv = {k: claimed[k] for k in cands if k in claimed}
        if not cands or len(cv) < len(cands):
            return None
        best = max(cv, key=cv.get)
        if list(cv.values()).count(cv[best]) > 1:
            return None
        return best
    if skill == "compare_count":
        cands = list(options) if options else list(gold_q or {})
        if len(cands) != 2 or any(k not in claimed for k in cands):
            return None
        a, b = cands
        if claimed[a] == claimed[b]:
            return None
        return a if claimed[a] > claimed[b] else b
    if skill == "count_band":
        key = _resolve_key(q, avail)
        if key is None:
            return None
        lo, hi = _parse_bands(q)
        c = float(claimed.get(key, 0.0))
        return "few" if c <= lo else ("moderate" if c <= hi else "many")
    if skill == "yesno":
        ql = q.lower()
        if any(s in ql for s in ("more numerous", "more than", "outnumber")):
            if " than " not in ql:
                return None
            left, right = q.split(" than ", 1)
            ka, kb = _resolve_key(left, avail), _resolve_key(right, avail)
            if not ka or not kb:
                return None
            ca, cb = float(claimed.get(ka, 0.0)), float(claimed.get(kb, 0.0))
            if ca == cb:
                return None
            return "yes" if ca > cb else "no"
        if any(s in ql for s in ("predominant", "mostly", "mainly", "most abundant")):
            key = _resolve_key(q, avail)
            if key is None or not claimed:
                return None
            best = max(claimed, key=claimed.get)
            if list(claimed.values()).count(claimed[best]) > 1:
                return None
            return "yes" if best == key else "no"
        key = _resolve_key(q, avail)
        if key is None:
            return None
        return "yes" if float(claimed.get(key, 0.0)) > 0 else "no"
    return None


def reexecute(claimed, skill, gold_q, target, options=None, opt_map=None,
              cons_mode="auto", question=None):
    if target is None:
        return None
    if skill in COUNTING:
        return consistency(claimed, skill, gold_q, target, opt_map=opt_map,
                           mode=cons_mode)
    fc = _fofC_s1(claimed, skill, gold_q, options, question)
    return None if fc is None else float(fc == target)


def decompose(text, gold_q, skill, gold_ans, options=None, opt_map=None,
              cons_mode="auto", question=None, kappa=KAPPA, pred_override=None):
    claimed = extract_quantities(text)
    pred = pred_override if pred_override is not None else _extract_pred(text, options)
    vs, flags = struct_valid(text, options)

    A = float(pred is not None and gold_ans is not None and pred == gold_ans)
    S, n_ok, n_req = step_score(claimed, gold_q)
    p, _ = smooth_p(claimed, gold_q, kappa)
    B = reexecute(claimed, skill, gold_q, gold_ans, options, opt_map, cons_mode, question)
    D = reexecute(claimed, skill, gold_q, pred, options, opt_map, cons_mode, question)

    lo_keys = [k for k, v in (gold_q or {}).items() if abs(float(v)) <= 5]
    lo_hit = (sum(1 for k in lo_keys
                  if claimed and abs(float(claimed.get(k, 0.0)) - float(gold_q[k])) < 1e-6)
              / len(lo_keys)) if lo_keys else None

    return {
        "A": A, "S": S, "p": p,
        "B": (None if B is None else float(B)),
        "D": (None if D is None else float(D)),
        "V_s": float(vs), "flags": flags,
        "n_ok": n_ok, "n_req": n_req,
        "claimed": claimed is not None,
        "matched": bool(claimed) and bool(gold_q) and set(gold_q) <= set(claimed or {}),
        "pred": pred, "lo_hit": lo_hit,
    }


