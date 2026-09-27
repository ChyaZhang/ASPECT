import re


def parse_pred(reply, opts):
    if not reply:
        return None
    m = list(re.finditer(r"FINAL\s*:\s*(.+)", reply, re.I))
    tail = (m[-1].group(1) if m else reply).strip().strip("`*\"'.<> ")
    low = tail.lower()
    for o in opts:
        if low == o.lower():
            return o
    for o in sorted(opts, key=len, reverse=True):
        if re.search(r"(?<![\w:])" + re.escape(o.lower()) + r"(?![\w:])", low):
            return o
    for o in sorted(opts, key=len, reverse=True):
        if reply.lower().rfind(o.lower()) >= 0:
            return o
    return None
