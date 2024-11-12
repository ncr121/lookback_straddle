# Lookback Straddle Signals
This project computes the momentum signal using lookback straddles, which is an alternative to the usual moving average crossover signals approach (e.g. 50 day and 200 days) which does not provide the strength of the signal analytically. 
## Description
Core functions used in the backtesting of a momentum strategy and also combined with other signals for a composite strategy.
The two main functions contained in this code are:
compute_avg_straddle_delta_final  - this is an important function that determines the magnitude and the direction of the trend-following signal. There is theoretical justification for a lookback straddle as a proxy for a momentum strategy (Fung & Hsieh 2001; this function calculates the aggregate delta of 4 options (ATM calls and puts) using the Black-Scholes model. Specifically, we look at 2 lookback straddles, one with the highest up-strike in the lookback period, and the other a down-strike straddle. 
compute_ATR - this is a simple routine to calc the average true range of a series which gives a higher frequency and more accurate measure of volatility compared to daily vol. The previous day's close is compared to the high, low and close of the latest day's price. 
