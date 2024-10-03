import datetime as dt
import numpy as np
import pandas as pd
import mibian

import utilities_Ganesh as util


def get_bbg_historical(bbg, securities, fields, start_date, end_date, periodicity='D'):
    return bbg.get_historical(securities, fields, start_date, end_date, index_type=dt.date, period=periodicity)


def get_bbg_data(bbg, df_config, ticker, fields, start_date, end_date, periodicity='D'):
    """This function gets the price data series from bloomberg."""
    df = get_bbg_historical(bbg, df_config[ticker], fields, start_date, end_date, periodicity).droplevel('Field', axis=1)

    return df[df_config[ticker]].set_axis(df_config.index, axis=1).squeeze()


def compute_avg_straddle_delta_final(risk_free_rates):
    def compute_avg_straddle_delta_final(df_factor):
        """Computes the lookback straddle for the strategy."""
        current_date = df_factor.index[-1]
        spot = df_factor.loc[current_date]  # spot = df_factor.iloc[-1]
        vol = df_factor.pct_change().std() * np.sqrt(252) * 100
    
        risk_free_rate = risk_free_rates[current_date]
        total_delta = 0
    
        for day, date_idx in enumerate(df_factor.index[:-1], start=1):
            up_strike = df_factor.loc[date_idx:].max()
            down_strike = df_factor.loc[date_idx:].min()
    
            up_straddle = mibian.BS([spot, up_strike, risk_free_rate, day], volatility=vol)
            up_call_delta = up_straddle.callDelta
            up_put_delta = up_straddle.putDelta
            down_straddle = mibian.BS([spot, down_strike, risk_free_rate, day], volatility=vol)
            down_call_delta = down_straddle.callDelta
            down_put_delta = down_straddle.putDelta
            straddle_delta = up_call_delta + up_put_delta + down_call_delta + down_put_delta
            total_delta += straddle_delta
    
        return total_delta / len(df_factor)

    return compute_avg_straddle_delta_final


def compute_ATR(daily_prices, df_config):
    """This function computes the exponential average true range for each asset
    based on a 60 day lookback period and using the open high low close prices for the asset."""
    daily_close = daily_prices['PX_LAST'].shift(1)

    return pd.concat([pd.DataFrame([daily_prices['PX_HIGH'][asset] - daily_prices['PX_LOW'][asset],
                                    (daily_prices['PX_HIGH'][asset] - daily_close[asset]).abs(),
                                    (daily_prices['PX_LOW'][asset] - daily_close[asset]).abs()]
                                   ).max().T.ewm(span=df_config.loc[asset, 'TREND_ATR_PERIOD']).mean()
                      for asset in df_config.index], axis=1, keys=df_config.index)


def break_data_per_asset(component, db, classes, dict_class):  
    """This function seggregates a given dataframe into individual dataframes
    based on the each individual asset class in the strategy."""
    for asset_class in classes:        
        db[asset_class][component] = db['Cross_Asset'][component][dict_class[asset_class]]


def check_changein_signage_risk_thresh(df_fut_monthly, df_GBP_ATR, trend_rebal, risk_target):
    """For each rebalancing date this function checks whether the change from previous rebal to today
    is beyond a threshold or not."""
    df_new_monthly = pd.DataFrame().reindex_like(df_fut_monthly)
    for index, row in df_fut_monthly.iterrows():
        if index == df_fut_monthly.index[0]:
            df_new_monthly.loc[index] = df_fut_monthly.loc[index]
        else:
            prev_date_loc = df_fut_monthly.index.get_loc(index) - 1
            df_new_monthly.loc[index] = df_new_monthly.iloc[prev_date_loc]

            sign = pd.DataFrame(0, ['OLD','NEW'], df_fut_monthly.columns)
            risk = pd.DataFrame(0, ['OLD','NEW'], df_fut_monthly.columns)
            
            risk.loc['OLD']             = (df_new_monthly.iloc[prev_date_loc]*df_GBP_ATR.loc[index]).abs()
            risk.loc['NEW']             = (df_fut_monthly.loc[index]*df_GBP_ATR.loc[index]).abs()

            sign.loc['NEW'][df_fut_monthly.loc[index] > 0]  =  1
            sign.loc['NEW'][df_fut_monthly.loc[index] < 0]  = -1
            
            sign.loc['OLD'][df_new_monthly.iloc[prev_date_loc] > 0]  =  1
            sign.loc['OLD'][df_new_monthly.iloc[prev_date_loc] < 0]  = -1 
            
            sign_check                  = sign.loc['NEW'] != sign.loc['OLD']
            target_vol = pd.DataFrame(risk_target, df_GBP_ATR.index , df_GBP_ATR.columns)
            target_vol                  = target_vol[df_fut_monthly.columns]

            risk_check_UB               = risk.loc['OLD'] >= target_vol.loc[index]*( 1 + trend_rebal)
            risk_check_LB               = risk.loc['OLD'] <= target_vol.loc[index]*( 1 - trend_rebal)
            risk_check                  = risk_check_UB | risk_check_LB
            
            combined_check              = sign_check | risk_check

            df_new_monthly.loc[index][combined_check] = df_fut_monthly.loc[index][combined_check]

    return df_new_monthly


def compute_weights_BPS_scheme(df_fut_monthly_adj, scheme_value, fx_price, no_adj_prices, df_config): 
    """Given the total futures contract per asset and the total fund value,
    this function computes in %, the total weight of fund value allocated to each asset."""
    fx_price_monthly = fx_price.reindex(df_fut_monthly_adj.index)
    no_adj_prices_monthly = no_adj_prices.reindex(df_fut_monthly_adj.index)
    multipliers = df_config['FUT Multiplier'] * df_config['FX Multiplier']

    return df_fut_monthly_adj * fx_price_monthly * no_adj_prices_monthly * multipliers / scheme_value


def run_strategy(asset_class, assets, db, monthly_returns, look_back_rolling_sharpe, periods_per_year, annualised, statistics_monthly):
    """Calls individual backtest function from the backtest engine to run performance statistics
    for the strategy."""
    strategy = asset_class + ' ADJ'
    df_rets = monthly_returns[assets]
    return_series = util.getResults(db['Cross_Asset']['BPS_Weights_Adj'][assets], df_rets)
    db['Strategy Returns'][strategy] = return_series
    return_series = return_series[return_series.ne(0).idxmax():]
    results, sharpe_roll_wind = util.summaryStatistics(return_series, 0, strategy, look_back_rolling_sharpe, periods_per_year, annualised)
    util.prntResults(results, return_series, monthly_returns, sharpe_roll_wind, assets, statistics_monthly, strategy ,look_back_rolling_sharpe)    
    








