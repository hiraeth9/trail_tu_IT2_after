from .it2_trust import trust_computeT2_new
def self_trust_from_counts(succ_timely: float, succ_delay: float, fail: float) -> float:
    # 统一转浮点，避免整型除法坑
    st = float(succ_timely)
    sd = float(succ_delay)
    fl = float(fail)

    succ = max(0.0, st + sd)
    # DPR = fail / (succ + fail)
    dpr = fl / max(1.0, succ + fl)

    # DLR = succ_delay / succ_timely（若无及时成功，则定义为最差 1.0）
    dlr = (sd / st) if st > 0.0 else 1.0

    # 夹到 [0,1]
    dpr = min(max(dpr, 0.0), 1.0)
    dlr = min(max(dlr, 0.0), 1.0)

    L, R = trust_computeT2_new(dpr, dlr)
    return float((L + R) / 2.0)
