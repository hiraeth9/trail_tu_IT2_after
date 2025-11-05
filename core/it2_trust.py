def trust_computeT2_new(DPR, DLR):
    trust_average = []
    grade_U = []
    grade_L = []

    # rule1
    grade = low_DLR(DLR) * low_DPR(DPR)
    temp_trust, temp_gU, temp_gL = com_T(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU)
    grade_L.append(temp_gL)

    # rule2
    grade = med_DLR(DLR) * low_DPR(DPR)
    temp_trust, temp_gU, temp_gL = T_L(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU / 2)
    grade_L.append(temp_gL / 2)

    temp_trust, temp_gU, temp_gL = T_R(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU / 2)
    grade_L.append(temp_gL / 2)

    # rule3
    grade = high_DLR(DLR) * low_DPR(DPR)
    temp_trust, temp_gU, temp_gL = med_T_L(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU / 2)
    grade_L.append(temp_gL / 2)

    temp_trust, temp_gU, temp_gL = med_T_R(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU / 2)
    grade_L.append(temp_gL / 2)

    # rule4
    grade = low_DLR(DLR) * med_DPR(DPR)
    temp_trust, temp_gU, temp_gL = med_T_L(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU / 2)
    grade_L.append(temp_gL / 2)

    temp_trust, temp_gU, temp_gL = med_T_R(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU / 2)
    grade_L.append(temp_gL / 2)

    # rule5
    grade = med_DLR(DLR) * med_DPR(DPR)
    temp_trust, temp_gU, temp_gL = med_DT_L(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU / 2)
    grade_L.append(temp_gL / 2)

    temp_trust, temp_gU, temp_gL = med_DT_R(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU / 2)
    grade_L.append(temp_gL / 2)

    # rule6
    grade = high_DLR(DLR) * med_DPR(DPR)
    temp_trust, temp_gU, temp_gL = DT_L(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU / 2)
    grade_L.append(temp_gL / 2)

    temp_trust, temp_gU, temp_gL = DT_R(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU / 2)
    grade_L.append(temp_gL / 2)

    # rule7
    grade = low_DLR(DLR) * high_DPR(DPR)
    temp_trust, temp_gU, temp_gL = DT_L(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU / 2)
    grade_L.append(temp_gL / 2)

    temp_trust, temp_gU, temp_gL = DT_R(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU / 2)
    grade_L.append(temp_gL / 2)

    # rule8
    grade = med_DLR(DLR) * high_DPR(DPR)
    temp_trust, temp_gU, temp_gL = int_DT_L(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU / 2)
    grade_L.append(temp_gL / 2)

    temp_trust, temp_gU, temp_gL = int_DT_R(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU / 2)
    grade_L.append(temp_gL / 2)

    # rule9
    grade = high_DLR(DLR) * high_DPR(DPR)
    temp_trust, temp_gU, temp_gL = com_DT(grade)
    trust_average.append(temp_trust)
    grade_U.append(temp_gU)
    grade_L.append(temp_gL)
    # trust output
    # normalization for grade_Lower and grade_Upper
    for i in range(len(grade_U)):
        grade_U[i] = grade_U[i] / sum(grade_U)
        grade_L[i] = grade_L[i] / sum(grade_L)

    trust_Left = trust_average
    trust_Right = trust_average
    grade_Lower = grade_L
    grade_Upper = grade_U
    # calculate the left trust: trust_L
    trust_Left.sort()
    trust_Right.sort()
    grade_Lower.sort()
    grade_Upper.sort()
    #
    ha = 0
    hb = 0
    hL = 0
    for i in range(len(trust_Left)):
        ha += grade_Lower[i] * trust_Left[i]
        hb += grade_Lower[i]
    for i in range(len(trust_Left)):
        hL += 1
        ha += trust_Left[hL] * (grade_Upper[hL] - grade_Lower[hL])
        hb += grade_Upper[hL] - grade_Lower[hL]
        trust_L = ha / hb
        if hL >= len(trust_Left):
            break
        if trust_L <= trust_Left[hL + 1]:
            break

    trust_Left.sort()
    trust_Right.sort()
    grade_Lower.sort()
    grade_Upper.sort()
    ha = 0
    hb = 0
    hR = 9
    for i in range(len(trust_Left)):
        ha += grade_Lower[i] * trust_Right[i]
        hb += grade_Lower[i]

    for i in range(len(trust_Left)):
        ha += trust_Right[hR] * (grade_Upper[hR] - grade_Lower[hR])
        hb += grade_Upper[hR] - grade_Lower[hR]
        trust_R = ha / hb
        hR -= 1

        if hR <= 0:
            break
        if trust_R >= trust_Right[hR]:
            break
    return trust_L, trust_R


# 输入DPR的模糊集
def low_DPR(x):
    if x <= 0:
        return 1
    elif x <= 0.4:
        return (4 - 10 * x) / 4
    else:
        return 0


def med_DPR(x):
    if x <= 0:
        return 0
    elif x <= 0.4:
        return (10 * x - 0) / 4
    elif x <= 0.8:
        return (8 - 10 * x) / 4
    else:
        return 0


def high_DPR(x):
    if x <= 0.4:
        return 0
    elif x <= 0.8:
        return (10 * x - 4) / 4
    else:
        return 1


# 输入模糊集 DLR
def low_DLR(x):
    if x <= 0:
        return 1
    elif x <= 0.5:
        return (5 - 10 * x) / 5
    else:
        return 0


def med_DLR(x):
    if x <= 0:
        return 0
    elif x <= 0.5:
        return 10 * x / 5
    elif x <= 1:
        return (10 - 10 * x) / 5
    else:
        return 0


def high_DLR(x):
    if x <= 0.5:
        return 0
    elif x <= 1:
        return (10 * x - 5) / 5
    else:
        return 1


# output fuzzy set membership function
# complete distrust
def com_DT(y):
    if y >= 1:
        x = (0 + 0.05) / 2
        gU = 1
        gL = 1
    elif y > 0:
        x = ((0.2 - 0.15 * y) + (0.15 - 0.1 * y)) / 2
        gU = (0.2 - x) / 0.15
        gL = (0.15 - x) / 0.1
        if gL < 0:
            gL = 0
    else:
        x = 0
        gU = 0
        gL = 0
    return x, gU, gL


def int_DT_R(y):
    if y >= 1:
        x = 0.2
        gU = 1
        gL = 1
    elif y > 0:
        x = ((0.35 - 0.15 * y) + (0.3 - 0.1 * y)) / 2
        gU = (0.35 - x) / 0.15
        gL = (0.3 - x) / 0.1
        if gL < 0:
            gL = 0
    else:
        x = 0
        gU = 0
        gL = 0
    return x, gU, gL


def int_DT_L(y):
    if y >= 1:
        x = 0.2
        gU = 1
        gL = 1
    elif y > 0:
        x = ((0.15 * y + 0.05) + (0.1 * y + 0.1)) / 2
        gU = (x - 0.05) / 0.15
        gL = (x - 0.1) / 0.1
        if gL < 0:
            gL = 0
    else:
        x = 0
        gU = 0
        gL = 0
    return x, gU, gL


def DT_R(y):
    if y >= 1:
        x = 0.35
        gU = 1
        gL = 1
    elif y > 0:
        x = ((0.5 - 0.15 * y) + (0.45 - 0.1 * y)) / 2
        gU = (0.5 - x) / 0.15
        gL = (0.45 - x) / 0.1
        if gL < 0:
            gL = 0
    else:
        x = 0
        gU = 0
        gL = 0
    return x, gU, gL


def DT_L(y):
    if y >= 1:
        x = 0.35
        gU = 1
        gL = 1
    elif y > 0:
        x = ((0.15 * y + 0.2) + (0.1 * y + 0.25)) / 2
        gU = (x - 0.2) / 0.15
        gL = (x - 0.25) / 0.1
        if gL < 0:
            gL = 0
    else:
        x = 0
        gU = 0
        gL = 0
    return x, gU, gL


def med_DT_R(y):
    if y >= 1:
        x = 0.5
        gU = 1
        gL = 1
    elif y > 0:
        x = ((0.65 - 0.15 * y) + (0.6 - 0.1 * y)) / 2
        gU = (0.65 - x) / 0.15
        gL = (0.6 - x) / 0.1
        if gL < 0:
            gL = 0
    else:
        x = 0
        gU = 0
        gL = 0
    return x, gU, gL


def med_DT_L(y):
    if y >= 1:
        x = 0.5
        gU = 1
        gL = 1
    elif y > 0:
        x = ((0.15 * y + 0.35) + (0.1 * y + 0.4)) / 2
        gU = (x - 0.35) / 0.15
        gL = (x - 0.4) / 0.1
        if gL < 0:
            gL = 0
    else:
        x = 0
        gU = 0
        gL = 0
    return x, gU, gL


def med_T_R(y):
    if y >= 1:
        x = 0.65
        gU = 1
        gL = 1
    elif y > 0:
        x = ((0.8 - 0.15 * y) + (0.75 - 0.1 * y)) / 2
        gU = (0.8 - x) / 0.15
        gL = (0.75 - x) / 0.1
        if gL < 0:
            gL = 0
    else:
        x = 0
        gU = 0
        gL = 0
    return x, gU, gL


def med_T_L(y):
    if y >= 1:
        x = 0.65
        gU = 1
        gL = 1
    elif y > 0:
        x = ((0.15 * y + 0.5) + (0.1 * y + 0.55)) / 2
        gU = (x - 0.5) / 0.15
        gL = (x - 0.55) / 0.1
        if gL < 0:
            gL = 0
    else:
        x = 0
        gU = 0
        gL = 0
    return x, gU, gL


def T_R(y):
    if y >= 1:
        x = 0.8
        gU = 1
        gL = 1
    elif y > 0:
        x = ((0.95 - 0.15 * y) + (0.9 - 0.1 * y)) / 2
        gU = (0.95 - x) / 0.15
        gL = (0.9 - x) / 0.1
        if gL < 0:
            gL = 0
    else:
        x = 0
        gU = 0
        gL = 0
    return x, gU, gL


def T_L(y):
    if y >= 1:
        x = 0.8
        gU = 1
        gL = 1
    elif y > 0:
        x = ((0.15 * y + 0.65) + (0.1 * y + 0.7)) / 2
        gU = (x - 0.65) / 0.15
        gL = (x - 0.7) / 0.1
        if gL < 0:
            gL = 0
    else:
        x = 0
        gU = 0
        gL = 0
    return x, gU, gL

def com_T(y):
    if y >= 1:
        x = (0.95 + 1) / 2
        gU = 1
        gL = 1
    elif y > 0:
        x = ((0.15 * y + 0.8) + (0.1 * y + 0.85)) / 2
        gU = (x - 0.8) / 0.15
        gL = (x - 0.85) / 0.1
        if gL < 0:
            gL = 0
    else:
        x = 0
        gU = 0
        gL = 0
    return x, gU, gL
