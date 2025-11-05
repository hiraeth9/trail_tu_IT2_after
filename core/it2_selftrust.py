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
def self_trust_from_incoming(in_map: dict) -> float:
    """
    in_map: {observer_id: [succ_timely, succ_delay, fail]}
    返回：按观测样本量加权的 IT2 自信任（别人对我的聚合）
    说明：不在这里做遗忘；遗忘应在调用处与 Beta 同节奏进行。
    """
    total_w, acc = 0.0, 0.0
    for _obs, buf in (in_map or {}).items():
        st, sd, fl = float(buf[0]), float(buf[1]), float(buf[2])
        w = st + sd + fl
        if w <= 0.0:
            continue
        ti = self_trust_from_counts(st, sd, fl)
        acc += ti * w
        total_w += w
    return (acc / total_w) if total_w > 0.0 else 0.5
