# Basic Framework for Backtesting of Systematic Strategies (Trend-Following)

This project provides a simple framework for calculating typical momentum signals used at CTAs (using both the standard moving average crossover signals and an alternative approach using lookback straddles). Different signals can be combined into a blended composite signal - currently, the composite signal consists of a slower lookback straddle signal and a faster breakout signal (50 day/200 day windows).  This same framework can easily be extended to handle value signals to create a more meaningful, diversified CTA type portfolio (with many more assets and different momentum, value, carry, volatility signals). Position sizes (weights) are based on an inverse ATR (average true range) measure, similar to inverse volatility scaling. The usual performance statistics including Sharpe Ratio, drawdown and duration of drawdowns are available).

# A brief rationale of the lookback straddle implementation as the trend signal
A standard moving average based trend signal, even with multiple windows, does not specify the strength of the signal(forecast) analytically. The advantage of the straddle framework is that it provides both the direction and the magnitude of the trend-following forecast.

This project computes the momentum signal using lookback straddles, which is an alternative to the usual moving average crossover signals approach (e.g. 50 day and 200 days) which does not provide the strength of the signal analytically. There is theoretical justification for a lookback straddle as a proxy for a momentum strategy (Fung & Hsieh 2001) by dynamically calculating the aggregate delta (typically Black-Scholes) of two lookback straddles at each point in time - one straddle with the highest up-strike in the relevant lookback period and the other straddle, with the lowest down-strike. So, on each day, the deltas of 4 different lookback options, two ATM calls and 2 ATM puts are combined - the aggregate delta is bounded in the [-1,+1] interval which is mapped to the strength of the buy or sell signal for that security.

## Description
Core functions used in the backtesting of a momentum strategy and also combined with other signals for a composite strategy.
The two main functions contained in this code are:

compute_avg_straddle_delta_window - this is an important function that determines the magnitude and the direction of the trend-following signal. There is theoretical justification for a lookback straddle as a proxy for a momentum strategy (Fung & Hsieh 2001; this function calculates the aggregate delta of 4 options (ATM calls and puts) using the Black-Scholes model. Specifically, we look at 2 lookback straddles, one with the highest up-strike in the lookback period, and the other a down-strike straddle.

compute_atr - this is a simple routine to calc the average true range of a series which gives a higher frequency and more accurate measure of volatility compared to daily vol. The previous day's close is compared to the high, low and close of the latest day's price. 

tHIS
