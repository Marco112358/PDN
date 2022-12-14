import pandas as pd
import numpy as np
import AMM_functions as ammf
import PDN_functions as pdnf
import plotly
import plotly.express as px
from datetime import datetime

# ASSUMES 50/50 CONSTANT PRODUCT AMM
# USER DEFINED INPUTS
price_ausd = 100  # price of token 'a' in USD
price_busd = 1  # = 50 .. price of token 'b' in USD... set this to 1 for a stablecoin pegged to USD
yld = 0.4  # expected percent yield from LP farming and from fees annualized
fee = 0.0  # this is the fee taken by the LP to swap tokens
r_a = 0.2  # borrowing rate of token A
r_b = 0.2  # borrowing rate of token B
supply_a_pdn = 0  # initial supply of token A
supply_b_pdn = 200  # initial supply of token B
leverage = 3.0  # initial leverage
mean_a = -0.2  # mean return used in brownian motion for token A
std_a = 1.0  # std used in brownian motion for token A
mean_b = 0.0  # mean return used in brownian motion for token B
std_b = 0.0  # std used in brownian motion for token B
days = 1  # this is the number of days since start of LP and the current price
no_periods = int(round(365 / days))
no_trials = 100  # I ran 1000 trials but it takes a while
trial_ex = 5  # this is just a random trial to use for some of the output graphs
dist = "norm"  # only normal distribution can be used

# Calculated Parameters based on user defined parameters above
lev_ratio_a = leverage / (2 * (leverage - 1))
price_ab = price_ausd / price_busd
borrowed_a = lev_ratio_a * (leverage - 1) * (supply_a_pdn + supply_b_pdn / price_ab)
borrowed_b = (1 - lev_ratio_a) * (leverage - 1) * (price_ab * supply_a_pdn + supply_b_pdn)
supply_a_tot = supply_a_pdn + borrowed_a
supply_b_tot = supply_b_pdn + borrowed_b
assets0 = price_ausd * supply_a_tot + price_busd * supply_b_tot
debt0 = price_ausd * borrowed_a + price_busd * borrowed_b
eq0 = assets0 - debt0

# Get the random prices
tm_st = datetime.now()
prc_a_tbl = ammf.rnd_prc_tbl(price_ausd, mean_a, std_a, days, no_trials, no_periods, dist)
prc_b_tbl = ammf.rnd_prc_tbl(price_busd, mean_b, std_b, days, no_trials, no_periods, dist)
tm_end = datetime.now()
print('time to create price tables is   ' + str(tm_end - tm_st) + ' seconds')

# Set up your final output tables
col_nms = ['Price A', 'Price B', 'No Rebalance', 'Daily Rebalance', 'Weekly Rebalance', '5% Threshold', '1% Threshold',
           'Hodl', 'Yield Farming']
pctls = pd.DataFrame(index={'mean', 'std', '1', '5', '10', '25', '50', '75', '90', '95', '99', 'rebals'}, columns=col_nms)
grph_tbl = pd.DataFrame(index=np.arange(0, int(no_periods / days) + 1), columns=col_nms)
trial_ex_tbl = pd.DataFrame(index=np.arange(0, int(no_periods / days) + 1), columns=col_nms)

# Loop through the strategies
for c, col in enumerate(col_nms[2:]):
    tm_st = datetime.now()
    # Set additional parameters based on the strategy you are simulating
    if col == 'No Rebalance':
        rebalance = 0  # 1 = yes rebalance, 0 = no rebalance
        rebal_type = 'period'  # 'period' or 'threshold'
        rebal_period = 1  # how many days between rebalance 1 = daily, 7 = weekly, etc.
        rebal_threshold = 0.1  # price change from previous balance at which you should rebalance
        strat = 'PDN'
    elif col == 'Daily Rebalance':
        rebalance = 1  # 1 = yes rebalance, 0 = no rebalance
        rebal_type = 'period'  # 'period' or 'threshold'
        rebal_period = 1  # how many days between rebalance 1 = daily, 7 = weekly, etc.
        rebal_threshold = 0.1  # price change from previous balance at which you should rebalance
        strat = 'PDN'
    elif col == 'Weekly Rebalance':
        rebalance = 1  # 1 = yes rebalance, 0 = no rebalance
        rebal_type = 'period'  # 'period' or 'threshold'
        rebal_period = 7  # how many days between rebalance 1 = daily, 7 = weekly, etc.
        rebal_threshold = 0.1  # price change from previous balance at which you should rebalance
        strat = 'PDN'
    elif col == '5% Threshold':
        rebalance = 1  # 1 = yes rebalance, 0 = no rebalance
        rebal_type = 'threshold'  # 'period' or 'threshold'
        rebal_period = 1  # how many days between rebalance 1 = daily, 7 = weekly, etc.
        rebal_threshold = 0.05  # price change from previous balance at which you should rebalance
        strat = 'PDN'
    elif col == '1% Threshold':
        rebalance = 1  # 1 = yes rebalance, 0 = no rebalance
        rebal_type = 'threshold'  # 'period' or 'threshold'
        rebal_period = 1  # how many days between rebalance 1 = daily, 7 = weekly, etc.
        rebal_threshold = 0.01  # price change from previous balance at which you should rebalance
        strat = 'PDN'
    elif col == 'Hodl':
        rebalance = 0  # 1 = yes rebalance, 0 = no rebalance
        rebal_type = 'threshold'  # 'period' or 'threshold'
        rebal_period = 1  # how many days between rebalance 1 = daily, 7 = weekly, etc.
        rebal_threshold = 0.1  # price change from previous balance at which you should rebalance
        strat = 'Hodl'
        lev_ratio_a = 0
        borrowed_a = 0
        borrowed_b = 0
        supply_a_tot = 0.5 * (supply_a_pdn * price_ausd + supply_b_pdn * price_busd) / price_ausd
        supply_b_tot = 0.5 * (supply_a_pdn * price_ausd + supply_b_pdn * price_busd) / price_busd
        assets0 = price_ausd * supply_a_tot + price_busd * supply_b_tot
        debt0 = price_ausd * borrowed_a + price_busd * borrowed_b
        eq0 = assets0 - debt0
    elif col == 'Yield Farming':
        rebalance = 0  # 1 = yes rebalance, 0 = no rebalance
        rebal_type = 'threshold'  # 'period' or 'threshold'
        rebal_period = 1  # how many days between rebalance 1 = daily, 7 = weekly, etc.
        rebal_threshold = 0.1  # price change from previous balance at which you should rebalance
        strat = 'Yield Farming'
        lev_ratio_a = 0
        borrowed_a = 0
        borrowed_b = 0
        supply_a_tot = 0.5 * (supply_a_pdn * price_ausd + supply_b_pdn * price_busd) / price_ausd
        supply_b_tot = 0.5 * (supply_a_pdn * price_ausd + supply_b_pdn * price_busd) / price_busd
        assets0 = price_ausd * supply_a_tot + price_busd * supply_b_tot
        debt0 = price_ausd * borrowed_a + price_busd * borrowed_b
        eq0 = assets0 - debt0
    else:
        rebalance = 0  # 1 = yes rebalance, 0 = no rebalance
        rebal_type = 'threshold'  # 'period' or 'threshold'
        rebal_period = 1  # how many days between rebalance 1 = daily, 7 = weekly, etc.
        rebal_threshold = 0.1  # price change from previous balance at which you should rebalance
        strat = 'PDN'
    # This function actually does all the legwork
    fin_pnl, fin_ave, rb_tbl = pdnf.sims_loop2(no_trials, no_periods, supply_a_tot, supply_b_tot, borrowed_a, borrowed_b,
                                              assets0, debt0, eq0, prc_a_tbl, prc_b_tbl, yld, r_a, r_b, days, leverage,
                                              rebalance, rebal_type, rebal_period, rebal_threshold, strat)
    # Set the percentiles and other final table stats
    pctls.loc['50', col] = np.percentile(fin_pnl.iloc[:, int(no_periods / days)], 50)
    pctls.loc['1', col] = np.percentile(fin_pnl.iloc[:, int(no_periods / days)], 1)
    pctls.loc['5', col] = np.percentile(fin_pnl.iloc[:, int(no_periods / days)], 5)
    pctls.loc['10', col] = np.percentile(fin_pnl.iloc[:, int(no_periods / days)], 10)
    pctls.loc['25', col] = np.percentile(fin_pnl.iloc[:, int(no_periods / days)], 25)
    pctls.loc['75', col] = np.percentile(fin_pnl.iloc[:, int(no_periods / days)], 75)
    pctls.loc['90', col] = np.percentile(fin_pnl.iloc[:, int(no_periods / days)], 90)
    pctls.loc['95', col] = np.percentile(fin_pnl.iloc[:, int(no_periods / days)], 95)
    pctls.loc['99', col] = np.percentile(fin_pnl.iloc[:, int(no_periods / days)], 99)
    pctls.loc['mean', col] = np.mean(fin_pnl.iloc[:, int(no_periods / days)])
    pctls.loc['std', col] = np.std(fin_pnl.iloc[:, int(no_periods / days)])
    pctls.loc['rebals', col] = np.mean(rb_tbl.iloc[:, 0])
    grph_tbl.loc[:, col] = fin_ave.iloc[:, 0]
    trial_ex_tbl.loc[:, col] = fin_pnl.loc[trial_ex, :]
    # just save the final output table so I could do some additional analysis at somepoint
    if col == 'No Rebalance':
        no_rb_fin = fin_pnl
    if col == 'Daily Rebalance':
        daily_rb_fin = fin_pnl
    if col == 'Weekly Rebalance':
        weekly_rb_fin = fin_pnl
    if col == '10% Threshold':
        tenpct_rb_fin = fin_pnl
    if col == '5% Threshold':
       fivepct_rb_fin = fin_pnl
    if col == 'Hodl':
        hld_fin = fin_pnl
    if col == 'Yield Farming':
        yf_pnl = fin_pnl

    tm_end = datetime.now()
    print('time to create ' + col + ' tables is ' + str(tm_end - tm_st) + ' seconds')

# Add the summary statistics for the prices
for c, col in enumerate(['Price A', 'Price B']):
    if col == 'Price A':
        prc_tbl = prc_a_tbl
    else:
        prc_tbl = prc_b_tbl
    pctls.loc['50', col] = np.percentile(prc_tbl.iloc[:, int(no_periods / days)], 50)
    pctls.loc['1', col] = np.percentile(prc_tbl.iloc[:, int(no_periods / days)], 1)
    pctls.loc['5', col] = np.percentile(prc_tbl.iloc[:, int(no_periods / days)], 5)
    pctls.loc['10', col] = np.percentile(prc_tbl.iloc[:, int(no_periods / days)], 10)
    pctls.loc['25', col] = np.percentile(prc_tbl.iloc[:, int(no_periods / days)], 25)
    pctls.loc['75', col] = np.percentile(prc_tbl.iloc[:, int(no_periods / days)], 75)
    pctls.loc['90', col] = np.percentile(prc_tbl.iloc[:, int(no_periods / days)], 90)
    pctls.loc['95', col] = np.percentile(prc_tbl.iloc[:, int(no_periods / days)], 95)
    pctls.loc['99', col] = np.percentile(prc_tbl.iloc[:, int(no_periods / days)], 99)
    pctls.loc['mean', col] = np.mean(prc_tbl.iloc[:, int(no_periods / days)])
    pctls.loc['std', col] = np.std(prc_tbl.iloc[:, int(no_periods / days)])
    pctls.loc['rebals', col] = 0
    grph_tbl.loc[:, col] = np.average(prc_tbl, axis=0)
    trial_ex_tbl.loc[:, col] = prc_tbl.loc[trial_ex, :]

pctls = pctls.reindex(index=['mean', 'std', '1', '5', '10', '25', '50', '75', '90', '95', '99', 'rebals'])
print(pctls)

# This is all to save my output locally and to run the plotly.express files
file_loc = 'C:/Users/ryanm/Documents/Excel/PDN output/'
pctls.to_csv(file_loc + 'pctls_mean' + str(mean_a) + '_std' + str(std_a) + '_trials' + str(no_trials) + '.csv')
trial_ex_tbl.to_csv(file_loc + 'trial_ex_tbl_mean' + str(mean_a) + '_std' + str(std_a) + '_trials' + str(no_trials)
                    + '.csv')
grph_tbl.to_csv(file_loc + 'grph_tbl_mean' + str(mean_a) + '_std' + str(std_a) + '_trials' + str(no_trials) + '.csv')

fig1 = px.line(data_frame=trial_ex_tbl.iloc[:, :1])
# fig1.write_image(file_loc + 'rand_trial_prca' + str(mean_a) + '_std' + str(std_a) + '_trials' + str(no_trials) +'.png')
fig1.write_html(file_loc + 'rand_trial_prca' + str(mean_a) + '_std' + str(std_a) + '_trials' + str(no_trials)
                + 'plot.html', auto_open=True)
# pl.offline.plot(fig1)
fig2 = px.line(data_frame=trial_ex_tbl.iloc[:, 2:])
fig2.write_html(file_loc + 'rand_trial_strats' + str(mean_a) + '_std' + str(std_a) + '_trials' + str(no_trials)
                + 'plot.html', auto_open=True)
fig3 = px.line(data_frame=grph_tbl.iloc[:, :1])
fig3.write_html(file_loc + 'aves_prca' + str(mean_a) + '_std' + str(std_a) + '_trials' + str(no_trials)
                + 'plot.html', auto_open=True)
fig4 = px.line(data_frame=grph_tbl.iloc[:, 2:])
fig4.write_html(file_loc + 'aves_strats' + str(mean_a) + '_std' + str(std_a) + '_trials' + str(no_trials)
                + 'plot.html', auto_open=True)

no_rb_fin2 = pd.DataFrame(data=no_rb_fin, dtype=float)
bad = no_rb_fin2.nsmallest(10, 365, keep='all').index
good = no_rb_fin2.nlargest(10, 365, keep='all').index

good_tbl_prc = prc_a_tbl.loc[good, :].T
bad_tbl_prc = prc_a_tbl.loc[bad, :].T
good_tbl_pnl = no_rb_fin.loc[good, :].T
bad_tbl_pnl = no_rb_fin.loc[bad, :].T
fig5 = px.line(data_frame=good_tbl_prc.iloc[:])
fig5.write_html(file_loc + 'good_norb_prca' + str(mean_a) + '_std' + str(std_a) + '_trials' + str(no_trials)
                + 'plot.html', auto_open=True)
fig6 = px.line(data_frame=bad_tbl_prc.iloc[:])
fig6.write_html(file_loc + 'bad_norb_prca' + str(mean_a) + '_std' + str(std_a) + '_trials' + str(no_trials)
                + 'plot.html', auto_open=True)
fig7 = px.line(data_frame=good_tbl_pnl.iloc[:])
fig5.write_html(file_loc + 'good_norb_strats' + str(mean_a) + '_std' + str(std_a) + '_trials' + str(no_trials)
                + 'plot.html', auto_open=True)
fig8 = px.line(data_frame=bad_tbl_pnl.iloc[:])
fig8.write_html(file_loc + 'bad_norb_strats' + str(mean_a) + '_std' + str(std_a) + '_trials' + str(no_trials)
                + 'plot.html', auto_open=True)
fig9 = px.histogram(prc_a_tbl.iloc[:, 365], nbins=100)
fig9.write_html(file_loc + 'prca_histogram' + str(mean_a) + '_std' + str(std_a) + '_trials' + str(no_trials)
                + 'plot.html', auto_open=True)
