import numpy as np
import scipy.stats as sc
import pandas as pd


def supply_new(supply_a=None, supply_b=None, gamma=None, e=None):
    supply_a_new = supply_a / np.sqrt(gamma * e)
    supply_b_new = (supply_a * supply_b) / supply_a_new
    return supply_a_new, supply_b_new


def data(price_ausd=None, price_busd=None, price_ausd_new=None, price_busd_new=None, supply_a=None, fee=None, RRa=0.5):
    price_ab = price_ausd / price_busd
    e = (price_ausd_new / price_busd_new) / price_ab
    supply_b = supply_a * price_ab * (1 - RRa) / RRa
    # k = supply_a * supply_b
    # price_ab_new = price_ab * e
    gamma = 1 + fee
    return price_ab, supply_b, e, gamma


def total_values(supply_a=None, supply_b=None, supply_a_new=None, supply_b_new=None, price_ausd_new=None,
                 price_busd_new=None, price_ausd_old=None, price_busd_old=None, days=1, yld=0):
    T_lp_new = supply_a_new * price_ausd_new + supply_b_new * price_busd_new
    T_lp_old = supply_a * price_ausd_old + supply_b * price_busd_old
    T_lp_ave = (T_lp_new + T_lp_old) / 2
    T_hodl = supply_a * price_ausd_new + supply_b * price_busd_new
    # IL_pct = (T_lp_new - T_hodl) / T_hodl
    T_lp_post_yld = T_lp_new + T_lp_ave * (np.exp(yld * days / 365) - 1)
    IL_pct_post_yld = (T_lp_post_yld - T_hodl) / T_hodl
    return IL_pct_post_yld, T_lp_post_yld, T_hodl


def supply_new_given_delta_b(delta_b=None, supply_b=None, supply_a=None, rra=0.5):
    delta_a = supply_a - (supply_a / (((supply_b + delta_b) / supply_b) ** ((1 - rra) / rra)))
    supply_a_new = supply_a - delta_a
    supply_b_new = supply_b + delta_b
    return supply_a_new, supply_b_new


def tv_delta_gamma_lev(supply_a=None, supply_b=None, price_ausd_new=None, price_busd_new=None, price_ausd_old=None,
                       price_busd_old=None, days=1, r_yf=0, borrowed_a=0, borrowed_b=0, r_a=0, r_b=0):
    # lending rates and yield farming compound continously
    # price moves linearly
    # Should equity be the first or 2nd version here?
    price_ab_new = price_ausd_new / price_busd_new
    price_ab_old = price_ausd_old / price_busd_old
    assets_old = (supply_a + borrowed_a) * price_ausd_old + (supply_b + borrowed_b) * price_busd_old
    assets = np.sqrt(price_ab_new) * ((supply_a + borrowed_a) * np.sqrt(price_ab_old) +
                                      (supply_b + borrowed_b) / np.sqrt(price_ab_old))
    debt = price_ab_new * borrowed_a * np.exp(r_a * days / 365) + borrowed_b * np.exp(r_b * days / 365)
    equity = assets - debt + ((assets + assets_old) / 2) * (np.exp(r_yf * days / 365) - 1)
    delta = 0.5 * price_ab_new ** (-0.5) * ((supply_a + borrowed_a) * np.sqrt(price_ab_old) +
                                            (supply_b + borrowed_b) / np.sqrt(price_ab_old)) - borrowed_a
    gamma = -0.25 * price_ab_new ** (-3 / 2) * ((supply_a + borrowed_a) * np.sqrt(price_ab_old) +
                                                (supply_b + borrowed_b) / np.sqrt(price_ab_old))
    return equity, delta, gamma


def lyf_out(supply_a_tot=None, supply_b_tot=None, price_ausd_new=None, price_busd_new=None, price_ausd_old=None,
            price_busd_old=None, days=1, r_yf=0, borrowed_a=0, borrowed_b=0, r_a=0, r_b=0):
    # lending rates and yield farming compound continously
    # price moves linearly
    # Should equity be the first or 2nd version here?
    price_ab_new = price_ausd_new / price_busd_new
    price_ab_old = price_ausd_old / price_busd_old
    assets_old = supply_a_tot * price_ausd_old + supply_b_tot * price_busd_old
    assets = np.sqrt(price_ab_new) * (supply_a_tot * np.sqrt(price_ab_old) + supply_b_tot / np.sqrt(price_ab_old))
    assets_fin = assets + ((assets + assets_old) / 2) * (np.exp(r_yf * days / 365) - 1)
    supply_a_new = assets_fin / 2 / price_ausd_new
    supply_b_new = assets_fin / 2 / price_busd_new
    borrowed_a_new = borrowed_a * np.exp(r_a * days / 365)
    borrowed_b_new = borrowed_b * np.exp(r_b * days / 365)
    debt = price_ab_new * borrowed_a_new + borrowed_b_new
    equity = assets_fin - debt
    return assets_fin, debt, equity, supply_a_new, supply_b_new, borrowed_a_new, borrowed_b_new


def stdnorm_rndm_var():
    # standard normal variable that should change every call
    x = sc.norm.rvs()
    return x


def lognorm_rndm_var(mean=0, std=1):
    # FAIL
    # ex = np.exp(mean + 0.5 * std ** 2)
    # varx = np.exp(2 * mean + std ** 2) * (np.exp(std ** 2) - 1)
    # varx = np.exp(2 * mean + 2 * std ** 2) - ex ** 2
    # logmean1 = np.log((mean ** 2) / ((mean ** 2 + std ** 2) ** 0.5))
    # logstd1 = np.log(1 + (std ** 2) / (mean ** 2))
    logstd = (np.log(1 + (std / (1 + mean)) ** 2)) ** 0.5
    # logmean = np.log(1 + mean) - (logstd ** 2) / 2
    x = sc.lognorm.rvs(logstd)
    return x


def prc_new(prc_prev=0, mean=0, std=0, step=1, days=1, dist="norm"):
    # Standard Normal Distribution used
    # Create a new wealth based on Brownian Motion off of Previous Wealth
    mean_fin = mean * days / 365
    std_fin = std / np.sqrt(365 / days)
    if dist == "norm":
        price_new = prc_prev * np.exp((mean_fin - 0.5 * (std_fin ** 2)) * step + std_fin * (step ** 2)
                                      * stdnorm_rndm_var())
    elif dist == "log":
        price_new = prc_prev * np.exp((mean_fin - 0.5 * (std_fin ** 2)) * step + std_fin * (step ** 2) *
                                      (lognorm_rndm_var(mean, std) - 1))
    return price_new


def rebalance_function(supply_a=None, supply_b=None, price_ausd_new=None, price_busd_new=None, borrowed_a=0, borrowed_b=0,
                       lev_max=3):
    assets = supply_a * price_ausd_new + supply_b * price_busd_new
    debt_a = borrowed_a * price_ausd_new
    debt_b = borrowed_b * price_busd_new
    debt_tot = debt_a + debt_b
    equity = assets - debt_tot
    assets_new = equity * lev_max
    debt_new = assets_new - equity
    lev_ratio_a = lev_max / (2 * (lev_max - 1))
    borrowed_a_rb = lev_ratio_a * debt_new / price_ausd_new
    borrowed_b_rb = (1 - lev_ratio_a) * debt_new / price_busd_new
    supply_a_rb = assets_new / 2 / price_ausd_new
    supply_b_rb = assets_new / 2 / price_busd_new
    # supply_a_chng = supply_a_rb - supply_a
    # supply_b_chng = supply_b_rb - supply_b
    # borrowed_a_chng = borrowed_a_rb - borrowed_a
    # borrowed_b_chng = borrowed_b_rb - borrowed_b

    # eq_out = pct_close * equity
    # aa2 = assets * (1 - pct_close) + eq_out
    # dd2 = debt_tot * (1 - pct_close)
    # assets_new = equity * lev_max
    # debt_new = assets_new - aa2 + dd2
    # borrowed_a_rb = lev_ratio_a * debt_new / price_ausd_new
    # borrowed_b_rb = (1 - lev_ratio_a) * debt_new / price_busd_new
    # supply_a_rb = assets_new / 2 / price_ausd_new
    # supply_b_rb = assets_new / 2 / price_busd_new
    return supply_a_rb, supply_b_rb, borrowed_a_rb, borrowed_b_rb


def rnd_prc_tbl(prc_st=None, mean=0, std=0, days=1, no_trials=1, no_periods=1, dist="norm"):
    fin = pd.DataFrame(index=np.arange(0, no_trials), columns=np.arange(0, int(no_periods / days) + 1))
    for n in np.arange(0, no_trials):
        fin.iloc[n, 0] = prc_st
        for t in np.arange(1, int(no_periods / days) + 1):
            if t == 1:
                fin.iloc[n, t] = prc_new(prc_st, mean, std, 1, days, dist)
            else:
                fin.iloc[n, t] = prc_new(fin.iloc[n, t - 1], mean, std, 1, days, dist)
    return fin


def hodl_out(supply_a_tot=None, supply_b_tot=None, price_ausd_new=None, price_busd_new=None):
    # lending rates and yield farming compound continously
    # price moves linearly
    # Should equity be the first or 2nd version here?
    assets = supply_a_tot * price_ausd_new + supply_b_tot * price_busd_new
    return assets

