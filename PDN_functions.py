import numpy as np
import pandas as pd
import AMM_functions as AMMf


def sims_loop(no_trials=1, no_periods=1, supply_a_tot=None, supply_b_tot=None, borrowed_a=None, borrowed_b=None,
              assets0=None, debt0=None, eq0=None, prc_a_tbl=None, prc_b_tbl=None, yld=0, r_a=0, r_b=0, days=1,
              leverage=3, rebalance=0, rebal_type=None, rebal_period=1, rebal_threshold=0.1, strat=None):
    # Set up final output tables
    fin_pnl = pd.DataFrame(index=np.arange(0, no_trials), columns=np.arange(0, int(no_periods / days) + 1))
    rebal_tbl = pd.DataFrame(index=np.arange(0, no_trials), columns={'Rebalances'})
    # loop through the number of trials
    for n in np.arange(0, no_trials):
        # this table is great for testing. shows you the entire period for 1 trial
        tbl = pd.DataFrame(index=np.arange(0, int(no_periods / days) + 1), columns=['A Price', 'B Price', 'A Supply', 'B Supply',
                                                                        'A Borrowed', 'B Borrowed', '% Price AB Change',
                                                                        'Assets (USD)', 'Debt (USD)', 'Equity (USD)',
                                                                        'Leverage', 'PnL (%)', 'Rebalance Indicator'])
        tbl.loc[0, :] = [prc_a_tbl.iloc[n, 0], prc_b_tbl.iloc[n, 0], supply_a_tot, supply_b_tot, borrowed_a, borrowed_b,
                         0, assets0, debt0, eq0, assets0 / eq0, 0, 0]
        # Set the initial time period based on the given information above
        rnd_price_ausd = prc_a_tbl.iloc[n, 1]
        rnd_price_busd = prc_b_tbl.iloc[n, 1]
        e = (rnd_price_ausd / rnd_price_busd) / (prc_a_tbl.iloc[n, 0] / prc_b_tbl.iloc[n, 0])
        # Hodl uses a different function
        if strat == 'Hodl':
            assets = AMMf.hodl_out(supply_a_tot, supply_b_tot, rnd_price_ausd, rnd_price_busd)
            supply_a_new = supply_a_tot
            supply_b_new = supply_b_tot
            borrowed_a_new = borrowed_a
            borrowed_b_new = borrowed_b
            debt = borrowed_a_new * rnd_price_ausd + borrowed_b_new * rnd_price_busd
            equity = assets - debt
        else:
            assets, debt, equity, supply_a_new, supply_b_new, borrowed_a_new, borrowed_b_new \
                = AMMf.lyf_out(supply_a_tot, supply_b_tot, rnd_price_ausd, rnd_price_busd, prc_a_tbl.iloc[n, 0],
                               prc_b_tbl.iloc[n, 0], days, yld, borrowed_a, borrowed_b, r_a, r_b)
        pnl = (equity - eq0) / eq0
        tbl.loc[1, :] = [rnd_price_ausd, rnd_price_busd, supply_a_new, supply_b_new, borrowed_a_new, borrowed_b_new,
                         e - 1, assets, debt, equity, assets / equity, pnl, 0]

        for i in np.arange(2, int(no_periods / days) + 1):
            # Loop through all other time periods (t+1 to n) if you do not want to rebalance
            rnd_price_ausd_new = prc_a_tbl.iloc[n, i]
            rnd_price_busd_new = prc_b_tbl.iloc[n, i]
            price_ausd_old = prc_a_tbl.iloc[n, i - 1]
            price_busd_old = prc_b_tbl.iloc[n, i - 1]
            e = (rnd_price_ausd_new / rnd_price_busd_new) / (price_ausd_old / price_busd_old)
            # This indexes the price of the last rebalance to check for the price threshold rebal strategy
            rb_list = [(x, y) for x, y in enumerate(tbl.loc[:, 'Rebalance Indicator'] == 1) if y is True]
            if not rb_list:
                rb_max_ind = 0
            else:
                rb_max_ind, _ = max(rb_list)
            # Hodl uses a different function
            if strat == 'Hodl':
                assets = AMMf.hodl_out(supply_a_tot, supply_b_tot, rnd_price_ausd_new, rnd_price_busd_new)
                supply_a_new = supply_a_tot
                supply_b_new = supply_b_tot
                borrowed_a_new = borrowed_a
                borrowed_b_new = borrowed_b
                debt = borrowed_a_new * rnd_price_ausd_new + borrowed_b_new * rnd_price_busd_new
                equity = assets - debt
                rb_ind = 0
            # These are the conditional statements to rebalance based on period to price change
            elif (rebalance == 1 and rebal_type == 'period' and (i / rebal_period).is_integer()) \
                    or (rebalance == 1 and rebal_type == 'threshold' and rebal_threshold <=
                        abs((price_ausd_old / price_busd_old) /
                            (tbl.loc[rb_max_ind, 'A Price'] / tbl.loc[rb_max_ind, 'B Price']) - 1)):
                # Rebalance
                supply_a_rb, supply_b_rb, borrowed_a_rb, borrowed_b_rb = \
                    AMMf.rebalance_function(tbl.loc[i - 1, 'A Supply'], tbl.loc[i - 1, 'B Supply'], price_ausd_old,
                                            price_busd_old,
                                            tbl.loc[i - 1, 'A Borrowed'], tbl.loc[i - 1, 'B Borrowed'], leverage)
                # Calculate your assets and equity
                assets, debt, equity, supply_a_new, supply_b_new, borrowed_a_new, borrowed_b_new \
                    = AMMf.lyf_out(supply_a_rb, supply_b_rb, rnd_price_ausd_new, rnd_price_busd_new, price_ausd_old,
                                   price_busd_old, days, yld, borrowed_a_rb, borrowed_b_rb, r_a, r_b)
                rb_ind = 1
            else:
                # Calculate your assets and equity for times when you don't rebalance
                assets, debt, equity, supply_a_new, supply_b_new, borrowed_a_new, borrowed_b_new \
                    = AMMf.lyf_out(tbl.loc[i - 1, 'A Supply'], tbl.loc[i - 1, 'B Supply'], rnd_price_ausd_new,
                                   rnd_price_busd_new, price_ausd_old, price_busd_old, days, yld,
                                   tbl.loc[i - 1, 'A Borrowed'],
                                   tbl.loc[i - 1, 'B Borrowed'], r_a, r_b)
                rb_ind = 0

            pnl = (equity - eq0) / eq0
            tbl.loc[i, :] = [rnd_price_ausd_new, rnd_price_busd_new, supply_a_new, supply_b_new, borrowed_a_new,
                             borrowed_b_new, e - 1, assets, debt, equity, assets / equity, pnl, rb_ind]
        fin_pnl.iloc[n, :] = tbl.iloc[:, 11]
        rebal_tbl.iloc[n, 0] = np.sum(tbl.loc[:, 'Rebalance Indicator'])
    fin_ave = pd.DataFrame(index=np.arange(0, int(no_periods / days) + 1), columns={'Average PnL (%)'})
    fin_ave.iloc[:, 0] = np.average(fin_pnl, axis=0)
    return fin_pnl, fin_ave, rebal_tbl


# USING ASSETS/DEBT AS THE REBALANCING TRIGGER
def sims_loop2(no_trials=1, no_periods=1, supply_a_tot=None, supply_b_tot=None, borrowed_a=None, borrowed_b=None,
              assets0=None, debt0=None, eq0=None, prc_a_tbl=None, prc_b_tbl=None, yld=0, r_a=0, r_b=0, days=1,
              leverage=3, rebalance=0, rebal_type=None, rebal_period=1, rebal_threshold=0.1, strat=None):
    # Set up final output tables
    fin_pnl = pd.DataFrame(index=np.arange(0, no_trials), columns=np.arange(0, int(no_periods / days) + 1))
    rebal_tbl = pd.DataFrame(index=np.arange(0, no_trials), columns={'Rebalances'})
    # loop through the number of trials
    for n in np.arange(0, no_trials):
        # this table is great for testing. shows you the entire period for 1 trial
        tbl = pd.DataFrame(index=np.arange(0, int(no_periods / days) + 1), columns=['A Price', 'B Price', 'A Supply', 'B Supply',
                                                                        'A Borrowed', 'B Borrowed', '% Price AB Change',
                                                                        'Assets (USD)', 'Debt (USD)', 'Equity (USD)',
                                                                        'Leverage', 'PnL (%)', 'Rebalance Indicator'])
        tbl.loc[0, :] = [prc_a_tbl.iloc[n, 0], prc_b_tbl.iloc[n, 0], supply_a_tot, supply_b_tot, borrowed_a, borrowed_b,
                         0, assets0, debt0, eq0, assets0 / eq0, 0, 0]
        debt_to_assets0 = debt0 / assets0
        # Set the initial time period based on the given information above
        rnd_price_ausd = prc_a_tbl.iloc[n, 1]
        rnd_price_busd = prc_b_tbl.iloc[n, 1]
        e = (rnd_price_ausd / rnd_price_busd) / (prc_a_tbl.iloc[n, 0] / prc_b_tbl.iloc[n, 0])
        # Hodl uses a different function
        if strat == 'Hodl':
            assets = AMMf.hodl_out(supply_a_tot, supply_b_tot, rnd_price_ausd, rnd_price_busd)
            supply_a_new = supply_a_tot
            supply_b_new = supply_b_tot
            borrowed_a_new = borrowed_a
            borrowed_b_new = borrowed_b
            debt = borrowed_a_new * rnd_price_ausd + borrowed_b_new * rnd_price_busd
            equity = assets - debt
        else:
            assets, debt, equity, supply_a_new, supply_b_new, borrowed_a_new, borrowed_b_new \
                = AMMf.lyf_out(supply_a_tot, supply_b_tot, rnd_price_ausd, rnd_price_busd, prc_a_tbl.iloc[n, 0],
                               prc_b_tbl.iloc[n, 0], days, yld, borrowed_a, borrowed_b, r_a, r_b)
        pnl = (equity - eq0) / eq0
        assets_old = assets
        equity_old = equity
        tbl.loc[1, :] = [rnd_price_ausd, rnd_price_busd, supply_a_new, supply_b_new, borrowed_a_new, borrowed_b_new,
                         e - 1, assets, debt, equity, assets / equity, pnl, 0]

        for i in np.arange(2, int(no_periods / days) + 1):
            # Loop through all other time periods (t+1 to n) if you do not want to rebalance
            rnd_price_ausd_new = prc_a_tbl.iloc[n, i]
            rnd_price_busd_new = prc_b_tbl.iloc[n, i]
            price_ausd_old = prc_a_tbl.iloc[n, i - 1]
            price_busd_old = prc_b_tbl.iloc[n, i - 1]
            e = (rnd_price_ausd_new / rnd_price_busd_new) / (price_ausd_old / price_busd_old)
            # This indexes the price of the last rebalance to check for the price threshold rebal strategy
            rb_list = [(x, y) for x, y in enumerate(tbl.loc[:, 'Rebalance Indicator'] == 1) if y is True]
            if not rb_list:
                rb_max_ind = 0
            else:
                rb_max_ind, _ = max(rb_list)
            # Hodl uses a different function
            if strat == 'Hodl':
                assets = AMMf.hodl_out(supply_a_tot, supply_b_tot, rnd_price_ausd_new, rnd_price_busd_new)
                supply_a_new = supply_a_tot
                supply_b_new = supply_b_tot
                borrowed_a_new = borrowed_a
                borrowed_b_new = borrowed_b
                debt = borrowed_a_new * rnd_price_ausd_new + borrowed_b_new * rnd_price_busd_new
                equity = assets - debt
                rb_ind = 0
            # These are the conditional statements to rebalance based on period to price change
            elif (rebalance == 1 and rebal_type == 'period' and (i / rebal_period).is_integer()) \
                    or (rebalance == 1 and rebal_type == 'threshold' and rebal_threshold <=
                        abs((assets_old / equity_old) / leverage - 1)):
                # Rebalance
                supply_a_rb, supply_b_rb, borrowed_a_rb, borrowed_b_rb = \
                    AMMf.rebalance_function(tbl.loc[i - 1, 'A Supply'], tbl.loc[i - 1, 'B Supply'], price_ausd_old,
                                            price_busd_old,
                                            tbl.loc[i - 1, 'A Borrowed'], tbl.loc[i - 1, 'B Borrowed'], leverage)
                # Calculate your assets and equity
                assets, debt, equity, supply_a_new, supply_b_new, borrowed_a_new, borrowed_b_new \
                    = AMMf.lyf_out(supply_a_rb, supply_b_rb, rnd_price_ausd_new, rnd_price_busd_new, price_ausd_old,
                                   price_busd_old, days, yld, borrowed_a_rb, borrowed_b_rb, r_a, r_b)
                rb_ind = 1
            else:
                # Calculate your assets and equity for times when you don't rebalance
                assets, debt, equity, supply_a_new, supply_b_new, borrowed_a_new, borrowed_b_new \
                    = AMMf.lyf_out(tbl.loc[i - 1, 'A Supply'], tbl.loc[i - 1, 'B Supply'], rnd_price_ausd_new,
                                   rnd_price_busd_new, price_ausd_old, price_busd_old, days, yld,
                                   tbl.loc[i - 1, 'A Borrowed'],
                                   tbl.loc[i - 1, 'B Borrowed'], r_a, r_b)
                rb_ind = 0
            assets_old = assets
            equity_old = equity
            pnl = (equity - eq0) / eq0
            tbl.loc[i, :] = [rnd_price_ausd_new, rnd_price_busd_new, supply_a_new, supply_b_new, borrowed_a_new,
                             borrowed_b_new, e - 1, assets, debt, equity, assets / equity, pnl, rb_ind]
        fin_pnl.iloc[n, :] = tbl.iloc[:, 11]
        rebal_tbl.iloc[n, 0] = np.sum(tbl.loc[:, 'Rebalance Indicator'])
    fin_ave = pd.DataFrame(index=np.arange(0, int(no_periods / days) + 1), columns={'Average PnL (%)'})
    fin_ave.iloc[:, 0] = np.average(fin_pnl, axis=0)
    return fin_pnl, fin_ave, rebal_tbl
