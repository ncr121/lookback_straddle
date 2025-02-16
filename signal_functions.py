import numpy as np
import pandas as pd
import scipy.stats as st
from functools import reduce

"""========================================================================="""
"""Miscellaneous functions"""


def compute_d1(s, k, r, t, v):
    """Vectorised implementation of the Black-Scholes model."""
    return (np.log(s / k) + np.outer(t, (r + v**2 / 2))) / (np.outer(np.sqrt(t), v))


def compute_deltas(d1):
    """Compute call and put deltas."""
    return st.norm.cdf(d1), -st.norm.cdf(-d1)


def compute_avg_straddle_delta_window(df, risk_free_rates, assets):
    """
    Lookback straddle calculated by aggregating the call and put deltas from
    the highest up-strike and the lowest down-strike using the Black-Scholes
    model.
    """
    s = df.iloc[-1]
    r = risk_free_rates[df.index[-1]] / 100
    v = df.pct_change().std() * np.sqrt(252)
    t = np.arange(1, 253) / 365

    ku = df[::-1].cummax()[::-1]
    kd = df[::-1].cummin()[::-1]
    deltas = [compute_deltas(compute_d1(s, k, r, t, v)) for k in [ku, kd]]
    ttl_deltas = sum(deltas[0]) + sum(deltas[1])
    avg_deltas = ttl_deltas.sum(axis=0) / (4 * len(df))

    return pd.Series(avg_deltas, assets, name=df.index[-1])


def compute_avg_straddle_delta(prices, lookback, risk_free_rates, assets, start_date):
    """Momentum signal using lookback straddles."""
    return pd.concat([compute_avg_straddle_delta_window(prices.iloc[i:i+lookback], risk_free_rates, assets)
                      for i in range(len(prices) - lookback + 1)
                      if prices.index[i + lookback - 1] >= start_date],
                     axis=1).T


def compute_discrete_straddle(signals, buffer):
    return pd.DataFrame(np.where(signals.abs() > buffer, np.sign(signals), 0), signals.index, signals.columns)


def elementwise_max(df1, df2):
    """Compute the elementwise max between two dataframes."""
    return pd.DataFrame(np.where(df1 >= df2, df1, df2), df1.index, df1.columns)


def compute_atr(prices, lookback):
    """
    Average true range over a lookback period. Provides a more complete risk
    measure than volatility by comparing the max and min of a given day 
    versus the previous day's close.
    """
    prev_close = prices['PX_LAST'].shift(1).fillna(0)

    return reduce(elementwise_max, [prices['PX_HIGH'] - prices['PX_LOW'],
                                    (prices['PX_HIGH'] - prev_close).abs(),
                                    (prices['PX_LOW'] - prev_close).abs()]
                  ).ewm(span=lookback).mean()


def apply_multipliers(signals, fx_prices, df_config):
    return signals * fx_prices * df_config['FUT Multiplier'] * df_config['FX Multiplier']


"""========================================================================="""
"""Signal functions"""


def moving_high_and_low(signals, signals_disc, prices, window1, window2, breakout):
    """
    Composite signal using a slower lookback straddle signal (200 day window)
    and a faster breakout signal (50 day window).
    """
    df1_high = prices.rolling(window1).max()
    df1_low = prices.rolling(window1).min()
    df2_high = prices.rolling(window2).max()
    df2_low = prices.rolling(window2).min()

    mask = pd.DataFrame().reindex_like(prices).fillna(0)

    for date in signals.index:
        prev_idx = mask.index.get_loc(date) - 1
        mask.loc[date] = mask.iloc[prev_idx]

        mask.loc[date][(prices.loc[date] >= df1_high.loc[date]) & (signals_disc.loc[date] == 1)] = 1
        mask.loc[date][(prices.loc[date] <= df1_low.loc[date]) & (signals_disc.loc[date] == -1)] = -1

        if breakout:
            mask.loc[date][(mask.iloc[prev_idx] == 1) &
                           ((signals_disc.loc[date] <= 0) | (prices.loc[date] <= df2_low.loc[date]))] = 0
            mask.loc[date][(mask.iloc[prev_idx] == -1) &
                         ((signals_disc.loc[date] >=  0) | (prices.loc[date] >= df2_high.loc[date]))] = 0

    return (mask.abs() * signals).reindex(signals.index)


"""========================================================================="""
"""Weight functions"""


def compute_bps_weights(prices, futures, fx_prices, df_config, scheme_value):
    return prices * apply_multipliers(futures, fx_prices, df_config) / scheme_value


def compute_unadj_weights(comp_signals, prices, fx_prices, df_config, atr, risk_target, scheme_value):
    """
    Similar to an inverse volatility (vol scaling) approach to size positions,
    an inverse ATR scaling is done to calculate "equal risk" weightings.
    """
    fut_unadj = np.floor((1 / atr) * risk_target * comp_signals)
    return compute_bps_weights(prices.reindex(fut_unadj.index), fut_unadj, fx_prices, df_config, scheme_value)


"""========================================================================="""
"""Backtesting functions"""


def compute_returns(returns, weights, comms=0):
    """
    Compute the weighted returns for each individual security as well as the
    portfolio returns across all securities.
    """
    weights1 = weights.shift(1).fillna(0)
    delta = weights - weights1
    cost = delta.abs() * comms
    weighted_returns = returns * weights1 - cost
    total_returns = weighted_returns.sum(axis=1)

    return weighted_returns, total_returns


def compute_drawdown(returns):
    """
    Compute the drawdown statistics of a return series. This includes drawdown
    and drawdown duration series as well as maximum drawdown and maximum
    drawdown duration.
    """
    df = returns.rename('Ret').to_frame()
    df['Cum Ret'] = np.cumprod(1 + df['Ret'])
    df['HWM'] = df['Cum Ret'].cummax()
    df['DD'] = (df['Cum Ret'] / df['HWM'] - 1).abs()
    df1 = (df['DD'] != 0).cumsum()
    df['DD Dur'] = df1 - df1.where(df['DD'] == 0).ffill()

    return {'dd': df, 'mdd': df['DD'].max(), 'mdd dur': int(df['DD Dur'].max()), 'terminal': df['Cum Ret'][-1]}


def performace(returns, weights, comms=0):
    """
    Analyse the performace of a strategy at a portfolio level by looking at
    statistics such as mean, standard deviation and Sharpe ratio.
    """
    asset_returns, total_returns = compute_returns(returns, weights, comms)

    stats = {'mean': total_returns.mean() * 252, 'std': total_returns.std() * np.sqrt(252),
             'skew': st.skew(total_returns), 'kurtosis': st.skew(total_returns)}
    stats['sharpe'] = stats['mean'] / stats['std']
    stats.update(compute_drawdown(total_returns))

    return asset_returns, stats


"""========================================================================="""
