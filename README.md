# Basic Framework for Backtesting of Systematic Strategies (Trend-Following)

This project provides a simple framework for calculating typical momentum signals used at CTAs (using both the standard moving average crossover signals and an alternative approach using lookback straddles). Different signals can be combined into a blended composite signal - currently, the composite signal consists of a slower lookback straddle signal and a faster breakout signal (50 day/200 day windows).  This same framework can easily be extended to handle value signals to create a more meaningful, diversified CTA type portfolio (with many more assets and different momentum, value, carry, volatility signals). Position sizes (weights) are based on an inverse ATR (average true range) measure, similar to inverse volatility scaling. The usual performance statistics including Sharpe Ratio, drawdown and duration of drawdowns are available).

# A brief rationale of the lookback straddle implementation as the trend signal
A standard moving average based trend signal, even with multiple windows, does not specify the strength of the signal(forecast) analytically. The advantage of the straddle framework is that it provides both the direction and the magnitude of the trend-following forecast.

This project computes the momentum signal using lookback straddles, which is an alternative to the usual moving average crossover signals approach (e.g. 50 day and 200 days) which does not provide the strength of the signal analytically. There is theoretical justification for a lookback straddle as a proxy for a momentum strategy (Fung & Hsieh 2001) by dynamically calculating the aggregate delta (typically Black-Scholes) of two lookback straddles at each point in time - one straddle with the highest up-strike in the relevant lookback period and the other straddle, with the lowest down-strike. So, on each day, the deltas of 4 different lookback options, two ATM calls and 2 ATM puts are combined - the aggregate delta is bounded in the [-1,+1] interval which is mapped to the strength of the buy or sell signal for that security.

## Description
# Data
Historical data is obtained via the Bloomberg API. The "RISK TICKER" field is passed to get the price series used in computing the signals.
# Straddle Signals
Lookback straddle calculated by aggregating the call and put deltas from the highest up-strike and the lowest down-strike using the Black-Scholes model.
# Composite Signals
Composite signal using a slower lookback straddle signal (200 day window) and a faster breakout signal (50 day window).
# Weights (Positions)
Similar to an inverse volatility (vol scaling) approach to size positions, an inverse ATR scaling is done to calculate "equal risk" weightings.
# Backtesting
