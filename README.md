# Implied Probability Density Functions the Structural Dynamics of Option-Imp lied Probability Distributions (IVPDF)

Overview

This project investigates the structural dynamics of option-implied probability distributions (IVPDFs), also known as risk-neutral densities (RNDs). These distributions represent the market’s forward-looking expectations of asset prices and contain valuable information on variance, skewness, and kurtosis, which map directly to perceived risk and tail-event probabilities.

By extracting and analyzing these densities, the project aims to:

- Link option market data to asset pricing theory.

- Quantify tail risk and shifts in investor sentiment around macroeconomic and geopolitical events.

- Analyze the informational content of RND moments for forecasting returns, volatility, and jump risk, also to see if there is the possibility of identifying abnormal behaviour in option markets in the vicininty of major events.

## Core Contributions

### 1. Theoretical Foundations

Risk-Neutral Density (RND): Derived from option prices under the no-arbitrage framework.

Breeden-Litzenberger Theorem: The second derivative of the call price curve with respect to strike yields the RND.

Key Distinction: Risk-neutral vs. physical probability distributions, linked via the stochastic discount factor (pricing kernel).

Interpolation & curve fitting of discrete option data.

Different pricing models (e.g., Black-Scholes, Heston, Bates) to understand if more efficient prices help better explain this phenomenon.

### 3. Informational Content of the RND

- Mean (1st moment): Market-expected Strike price

- Variance (2nd moment): Market-expected volatility.

- Skewness (3rd moment): Crash risk and downside fear.

- Kurtosis (4th moment): Jump probability and fat-tail risk.

Risk Premia:

- Volatility Risk Premium (VRP)

- Volatility Risk Premium (VRP)

- Skewness Risk Premium (SRP) — shown to predict excess returns and tail events.

### 4. Event-Driven Analysis

Novel focus: high-frequency RND dynamics around macroeconomic and geopolitical shocks.

Captures structural shifts in beliefs vs. risk aversion that aggregate measures like the VIX cannot detect.

Provides granular insights into how markets price uncertainty during critical announcements.

## Key References

- Breeden, D., & Litzenberger, R. (1978). Prices of state-contingent claims implicit in option prices.
- Bakshi, G., Kapadia, N., & Madan, D. (2003). Stock return characteristics, skew laws, and the differential pricing of individual equity options.
- Bollerslev, T., Tauchen, G., & Zhou, H. (2009). Expected stock returns and variance risk premia.

How to Use
1. Download the html file if you only want to see the result and the process behind it without running the code.

2. to run the code download the repository, and run either the Quarto file or the python script.

Future Directions

- Incorporating machine learning–based regularization for density estimation.
- Applying IVPDF methods to systemic risk monitoring.
- Forecasting movments using RND moment dynamics. like skewness risk premium.
