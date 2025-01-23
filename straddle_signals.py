import datetime as dt
import pandas as pd

import signal_functions as sigfn


def get_bbg_data(bbg, securities, fields, start_date, end_date, assets):
    df = bbg.get_historical(securities, fields, start_date, end_date, index_type=dt.date).droplevel('Field', axis=1)
    return df[securities.str.replace(':', '').str.replace('?', '')].set_axis(assets, axis=1)


"""========================================================================="""
"""Set manual inputs"""

config_file = 'TRH_Config_File.xlsx'
config_sheet = 'Config Final'

bbg_data_file = 'bbg_data.xlsx'

start_date_str = '2015-01-01'
end_date_str = None

assets = ['MSCI EM', 'CHINA', 'JAPAN', 'EUROSTOXX', 'USA','USA2','NASDAQ', 'BRENT CRUDE', 'GOLD', 'COPPER', 'BUND',
          'US 10YR', 'US LONG', 'GILT']

straddle_lookback = 252
straddle_buffer = 0.15

atr_looback = 252

comp_signal_fn = sigfn.moving_high_and_low
comp_signal_kwargs1 = {'window1': 100, 'window2': 50, 'breakout': True}
comp_signal_kwargs2 = {'prices': 'prices'}

weights_fn = sigfn.compute_unadj_weights
weights_kwargs1 = {'risk_target': 104812, 'scheme_value': 2.5e8}
weights_kwargs2 = {'prices': 'prices', 'fx_prices': 'fx_prices', 'df_config': 'df_config', 'atr': 'usd_atr'}

"""========================================================================="""
"""Read in Bloomberg data"""

start_date = pd.to_datetime(start_date_str).date()
end_date = dt.date.today() if end_date_str is None else pd.to_datetime(end_date_str)
bbg_start_date = start_date - dt.timedelta(400)

df_config = pd.read_excel(config_file, config_sheet, index_col=0).T.loc[assets]
df_config['START_DATE'].fillna(start_date, inplace=True)
_cols = ['FUT Multiplier', 'FX Multiplier', 'Comms']
df_config[_cols] = df_config[_cols].astype(float)

try:
    from data_connect.bbg_session import BloombergSession

    bbg = BloombergSession()

    risk_free_rates = bbg.get_historical('US0003M Index', 'PX_LAST', start_date, end_date, index_type=dt.date).squeeze()

    fx_prices = get_bbg_data(bbg, df_config['FX CODE'], 'PX_LAST', start_date, end_date, assets)

    print('Reading in daily prices from Bloomberg')
    daily_prices = {field: get_bbg_data(bbg, df_config['RISK_TICKER'], field, bbg_start_date, end_date, assets)
                    for field in ['PX_LAST', 'PX_HIGH', 'PX_LOW']}
except ModuleNotFoundError:
    print('Cannot connect to Bloomberg')

    risk_free_rates = pd.read_excel('bbg_data.xlsx', 'US0003M Index', index_col=0).squeeze()

    fx_prices = pd.read_excel('bbg_data.xlsx', 'FX_CODE', index_col=0)

    print('Reading in daily prices from Excel files')
    daily_prices = pd.read_excel('bbg_data.xlsx', ['PX_LAST', 'PX_HIGH', 'PX_LOW'], index_col=0)

    for _df in [risk_free_rates, fx_prices, *daily_prices.values()]:
        _df.index = _df.index.date

    start_date = risk_free_rates.index[0]
    end_date = risk_free_rates.index[-1]
finally:
    _dates = pd.date_range(start_date, end_date, freq='D')
    dates = _dates[_dates.weekday < 5]
    dates_monthly = dates.to_frame().resample('BM').last().index.date
    dates = dates.date

    prices = daily_prices['PX_LAST']

"""========================================================================="""
"""Compute signals"""

print('Computing straddle signals')
straddle_signals = sigfn.compute_avg_straddle_delta(prices, straddle_lookback, risk_free_rates, assets, start_date)
straddle_signals_discrete = sigfn.compute_discrete_straddle(straddle_signals, straddle_buffer)

atr = sigfn.compute_atr(daily_prices, atr_looback).reindex(dates)
usd_atr = sigfn.apply_multipliers(atr, fx_prices, df_config)

print('Computing composite signals')
comp_signals = comp_signal_fn(straddle_signals, straddle_signals_discrete, **comp_signal_kwargs1,
                              **{k: globals()[v] for k, v in comp_signal_kwargs2.items()})

print('Computing weights')
weights = weights_fn(comp_signals, **weights_kwargs1, **{k: globals()[v] for k, v in weights_kwargs2.items()})

"""========================================================================="""
"""Backtesting"""

daily_returns = prices.pct_change().reindex(dates)
asset_returns, portfolio_stats = sigfn.performace(daily_returns, weights)

"""========================================================================="""
