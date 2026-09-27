from __future__ import annotations

import json
import math
import re

LETTERS = "ABCDEFGHIJKL"

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*(?:</answer>|$)", re.I | re.S)
FINAL_RE = re.compile(r"FINAL\s*:\s*(.+)", re.I)
LETTER_ONLY_RE = re.compile(r"^\(?([A-L])[\.\)\s]*$", re.I)
OBSERVE_RE = re.compile(r"<observe>[^{}]*(\{.*?\})\s*</observe>", re.I | re.S)


def _strip_think(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.rsplit("</think>", 1)[-1] if "</think>" in text else text


def extract_answer(text: str, options: list[str] | None = None):
    if not isinstance(text, str):
        return None
    tail = _strip_think(text)
    cand = None
    m = ANSWER_TAG_RE.search(tail) or FINAL_RE.search(tail)
    if m:
        cand = m.group(1).strip()
    else:
        lines = [l.strip() for l in tail.splitlines() if l.strip()]
        if lines:
            cand = lines[-1]
    if not cand:
        return None
    cand = cand.strip().strip("*`_ 。.")
    lm = re.match(r"^\(?([A-L])[\.\)]\s*(.+)$", cand, re.I)
    if lm and options:
        letter, rest = lm.group(1).upper(), lm.group(2).strip()
        if rest in options:
            return rest
        i = LETTERS.index(letter)
        return options[i] if i < len(options) else None
    if options and cand in options:
        return cand
    lm2 = LETTER_ONLY_RE.match(cand)
    if lm2 and options:
        i = LETTERS.index(lm2.group(1).upper())
        return options[i] if i < len(options) else None
    if options:
        low = cand.lower()
        hit = [o for o in options if o.lower() == low]
        if hit:
            return hit[0]
    return cand or None


def extract_quantities(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    m = OBSERVE_RE.search(_strip_think(text))
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    out = {}
    for k, v in obj.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and math.isfinite(v):
            out[str(k)] = float(v)
    return out


def _tol(true_v: float) -> float:
    return max(1.0, 0.10 * abs(float(true_v)))


def step_score(claimed: dict | None, gold_q: dict) -> tuple[float, int, int]:
    if not gold_q:
        return 0.0, 0, 0
    claimed = claimed or {}
    ok = 0
    for k, tv in gold_q.items():
        cv = claimed.get(k)
        if cv is None:
            continue
        d = abs(float(cv) - float(tv))
        tol = _tol(tv)
        if d <= tol:
            ok += 1
    return ok / len(gold_q), ok, len(gold_q)


def _band_from_ratio(a: float, b: float) -> str:
    if b == 0:
        return "much_more" if a > 0 else "comparable"
    r = a / b
    return "much_more" if r >= 2.0 else ("much_fewer" if r <= 0.5 else "comparable")


def consistency(claimed: dict | None, skill: str, gold_q: dict,
                pred_answer: str | None, opt_map: dict | None = None,
                mode: str = "auto") -> float | None:
    if mode == "off" or not claimed or not pred_answer:
        return None
    opt_map = opt_map or {}
    keys = list(gold_q)
    if skill in ("region_compare", "type_compare", "compartment_spatial"):
        if len(keys) < 2 or any(k not in claimed for k in keys[:2]):
            return None
        lab = _band_from_ratio(claimed[keys[0]], claimed[keys[1]])
        return float(opt_map.get(lab, lab) == pred_answer)
    if skill == "argmax_region":
        if any(k not in claimed for k in keys):
            return None
        best = max(keys, key=lambda k: claimed[k])
        if sum(1 for k in keys if claimed[k] == claimed[best]) > 1:
            return None
        return float(opt_map.get(best, best) == pred_answer)
    if skill == "multi_hop":
        reg = [k for k in keys if not k.startswith("_")]
        if any(k not in claimed for k in reg):
            return None
        best = max(reg, key=lambda k: claimed[k])
        if sum(1 for k in reg if claimed[k] == claimed[best]) > 1:
            return None
        return float(str(pred_answer).split(":")[0] == best)
    return None


EQ_AREA = {"q0", "q1", "q2", "q3",
           "upperleft", "upperright", "lowerleft", "lowerright"}


HALVES = {"top", "bottom", "left", "right"}


def consistency_mode(rec: dict) -> str:
    if rec["skill"] not in ("region_compare", "compartment_spatial"):
        return "auto"
    a, b = str(rec.get("region", "")).split("|")[:2]
    if "@" in a or "@" in b:
        return "off"
    eq = (a in EQ_AREA and b in EQ_AREA) or (a in HALVES and b in HALVES)
    return "auto" if eq else "off"
