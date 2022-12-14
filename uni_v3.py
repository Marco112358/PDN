import math
from random import random


# Standard Normal variate using Box-Muller transform.
def random_bm(mu, sigma):
    u = 0
    v = 0
    while (u == 0):
        u = random()  # Converting [0,1) to (0,1)
    while (v == 0):
        v = random()
    mag = sigma * math.sqrt(-2.0 * math.log(u))
    return mag * math.cos(2.0 * math.pi * v) + mu


def calcImpLoss(lowerLimit, upperLimit, px, alpha):
    r = math.sqrt(upperLimit / lowerLimit)
    a1 = (math.sqrt(r) - px)
    a2 = (math.sqrt(r) / (math.sqrt(r) - 1)) * (2 * math.sqrt(px) * math.exp(alpha) - (px + 1))
    a3 = (math.sqrt(r) * px - 1)
    if (px < 1 / r):
        return a3
    elif (px > r):
        return a1

    return a2


def calcExpImpLoss(rangePerc, mu, sigma, alpha):
    upperPx = 1 + rangePerc
    lowerPx = 1 / upperPx
    Vhsum = 0
    impLossSum = 0
    numTries = 10000
    for i in range(numTries):
        t = 1
        W = random_bm(0, 1) * math.sqrt(t - 0)
        X = (math.log(1 + mu) - 0.5 * math.pow(math.log(1 + sigma), 2)) * t + math.log(1 + sigma) * W
        _px = math.exp(X)
        Vhsum += 1 + _px
        impLossSum += calcImpLoss(lowerPx, upperPx, _px, alpha)

    return (impLossSum / numTries) / (Vhsum / numTries)


def calcIV(rangePerc, mu, alpha):
    delta = 0.0001
    loSigma = 0
    hiSigma = 10
    midSigma = (loSigma + hiSigma) / 2
    k_ = midSigma
    i_ = upperSigma
    j_ = lowerSigma
    kRet = calcExpImpLoss(rangePerc, mu, k_, alpha)  # midSigma midRet
    iRet = calcExpImpLoss(rangePerc, mu, i_, alpha)  # hiSigma loRet
    jRet = calcExpImpLoss(rangePerc, mu, j_, alpha)  # loSigma hiRet
    while iRet < jRet:
        if abs(kRet - 0) <= delta:
            break
        if 0 < kRet:
            jRet = kRet
            j_ = k_
        elif 0 > kRet:
            iRet = kRet
            i_ = k_

        k_ = (i_ + j_) / 2
        kRet = calcExpImpLoss(rangePerc, mu, k_, alpha)

    return k_