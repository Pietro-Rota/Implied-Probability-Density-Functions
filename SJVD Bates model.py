# SJVD Bates model 
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


#| include: false
#| eval: false
#| 
import QuantLib as ql

from pathlib import Path
file_path = Path("C:\\Users\\pietr\\OneDrive - City St George's, University of London\\Documents\\a2025-2026 BAYES\\Implied_Probability_Density_Functions\\AAPL_option_chain.xlsx")
options_df = pd.read_excel(file_path)

# -----------------------------
# 1. SET PARAMETERS
# -----------------------------
# These are the "economically meaningful" parameters you mentioned.
today = ql.Date(1, 1, 2023)
ql.Settings.instance().evaluationDate = today

maturity_date = today + ql.Period(1, ql.Years) # 1-year simulation
risk_free_rate = 0.05  # r
dividend_yield = 0.02  # q
initial_spot_price = 100.0
initial_variance = 0.09  # v0 (volatility = sqrt(0.09) = 30%)

# Stochastic Volatility (Heston) Parameters
kappa = 2.0    # Mean-reversion speed (κ). How fast volatility returns to its long-term level.
theta = 0.09   # Long-run variance (θ). The level volatility reverts to.
sigma = 0.4    # Volatility of volatility (σ). How volatile the volatility is.
rho = -0.7     # Correlation (ρ). Typically negative (when price goes down, volatility goes up).

# Jump Parameters (Merton)
jump_intensity = 5.0  # λ (lambda). How often jumps happen per year (e.g., 5 times a year on average).
jump_mean = -0.05     # μ_J. The average size of the jump. -0.05 = average 5% drop.
jump_stddev = 0.10    # δ. The standard deviation of the jump size.

# -----------------------------
# 2. SET UP THE BATES MODEL
# -----------------------------
# Create the calendar and day counter
calendar = ql.UnitedKingdom()  # CORRECTED: Remove the (m=1) parameter
day_counter = ql.Actual365Fixed()

# Create the spot curve (risk-free rate)
spot_curve = ql.FlatForward(today, risk_free_rate, day_counter)
spot_curve_handle = ql.YieldTermStructureHandle(spot_curve)

# Create the dividend curve
dividend_curve = ql.FlatForward(today, dividend_yield, day_counter)
dividend_curve_handle = ql.YieldTermStructureHandle(dividend_curve)

# Create the Bates process
bates_process = ql.BatesProcess(spot_curve_handle,
                                dividend_curve_handle,
                                ql.QuoteHandle(ql.SimpleQuote(initial_spot_price)),
                                initial_variance,
                                kappa, theta, sigma, rho,
                                jump_intensity, jump_mean, jump_stddev)

# -----------------------------
# 3. SIMULATE PATHS - CORRECTED VERSION
# -----------------------------
time_steps_per_year = 252  # Daily steps

# CORRECTION: Create TimeGrid object first
time_grid = ql.TimeGrid(1.0, time_steps_per_year)  # 1.0 year with daily steps

# CORRECTION: Use the proper constructor with 3 arguments
rng = ql.GaussianRandomSequenceGenerator(
    ql.UniformRandomSequenceGenerator(
        bates_process.factors() * (len(time_grid) - 1),  # Important: factors * (steps)
        ql.UniformRandomGenerator()
    )
)

# CORRECTION: Use 3-argument constructor
seq = ql.GaussianMultiPathGenerator(bates_process, time_grid, rng)

# Number of paths to simulate
num_paths = 3
paths = []
for i in range(num_paths):
    sample_path = seq.next()
    path = sample_path.value()
    # Extract the price path
    price_path = [path[0][t] for t in range(len(path[0]))]
    variance_path = [path[1][t] for t in range(len(path[1]))]
    paths.append((price_path, variance_path))

# -----------------------------
# 4. VISUALIZE THE RESULTS
# -----------------------------
# Create time axis
time_axis = np.linspace(0, 1, len(paths[0][0]))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot the price path
ax1.plot(time_axis, paths[0][1], linewidth=1)
ax1.plot(time_axis, paths[1][1], linewidth=1)
ax1.set_title('Simulated Stock Price Path using Bates (SVJD) Model')
ax1.set_ylabel('Price')
ax1.grid(True)

# Mark any significant jumps (changes > 5% in one day)
price_changes = np.diff(paths[0][0]) / paths[0][0][:-1]
jump_indices = np.where(np.abs(price_changes) > 0.05)[0]
for idx in jump_indices:
    ax1.axvline(x=time_axis[idx+1], color='red', linestyle='--', alpha=0.7)

# Plot the stochastic volatility path
volatility_path = np.sqrt(np.maximum(paths[0][1], 0)) * 100  # Ensure positive, convert to %
ax2.plot(time_axis, volatility_path, linewidth=1, color='orange')
ax2.set_title('Stochastic Volatility Path')
ax2.set_ylabel('Volatility (%)')
ax2.set_xlabel('Time (Years)')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Print some statistics
print(f"Initial Price: {paths[0][0][0]:.2f}")
print(f"Final Price: {paths[0][0][-1]:.2f}")
print(f"Number of significant jumps (>5%): {len(jump_indices)}")
print(f"Final Volatility: {volatility_path[-1]:.2f}%")


def simulate_bates_model(S0=100, T=1, dt=1/252, r=0.05, q=0.02, 
                        v0=0.09, kappa=2.0, theta=0.09, sigma=0.4, rho=-0.7,
                        lambd=5.0, mu_j=-0.05, delta_j=0.10, n_paths=1):
    """
    Simple Euler discretization of Bates (SVJD) model
    """
    n_steps = int(T/dt)
    
    # Initialize arrays
    S = np.zeros((n_paths, n_steps))
    v = np.zeros((n_paths, n_steps))
    
    S[:, 0] = S0
    v[:, 0] = v0
    
    # Generate correlated Brownian motions
    for i in range(n_paths):
        for t in range(1, n_steps):
            # Generate correlated random shocks
            Z1 = np.random.normal()
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal()
            
            # Stochastic volatility component
            v[i, t] = v[i, t-1] + kappa * (theta - max(v[i, t-1], 0)) * dt + \
                      sigma * np.sqrt(max(v[i, t-1], 0)) * np.sqrt(dt) * Z2
            v[i, t] = max(v[i, t], 0)  # Ensure non-negative variance
            
            # Jump component
            jump = 0
            jump_prob = lambd * dt
            if np.random.random() < jump_prob:
                jump_size = np.random.normal(mu_j, delta_j)
                jump = np.exp(jump_size) - 1
            
            # Stock price evolution
            S[i, t] = S[i, t-1] * np.exp((r - q - 0.5 * max(v[i, t-1], 0)) * dt + 
                                        np.sqrt(max(v[i, t-1], 0)) * np.sqrt(dt) * Z1 + jump)
    
    return S, v

# Run simulation
S, v = simulate_bates_model()

# Plot results
time_axis = np.linspace(0, 1, S.shape[1])
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_axis, S[0])
plt.title('Bates Model Simulation')
plt.ylabel('Price')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_axis, np.sqrt(v[0]) * 100)
plt.ylabel('Volatility (%)')
plt.xlabel('Time (Years)')
plt.grid(True)
plt.tight_layout()
plt.show()

class BatesPricer:
    """Simplified Bates model option pricer for calibration"""
    
    @staticmethod
    def heston_characteristic_function(u, T, S0, r, q, v0, kappa, theta, sigma, rho):
        """Heston model characteristic function"""
        # Simplified Heston characteristic function
        # In practice, you'd use the full complex formulation
        d = np.sqrt((rho * sigma * 1j * u - kappa)**2 + sigma**2 * (1j * u + u**2))
        g = (kappa - rho * sigma * 1j * u - d) / (kappa - rho * sigma * 1j * u + d)
        C = kappa * (theta * T - (2/sigma**2) * np.log((1 - g * np.exp(-d * T))/(1 - g)))
        D = ((kappa - rho * sigma * 1j * u - d) / sigma**2) * ((1 - np.exp(-d * T))/(1 - g * np.exp(-d * T)))
        
        return np.exp(1j * u * np.log(S0) + C + D * v0 + 1j * u * (r - q) * T)
    
    @staticmethod
    def bates_approximation(S0, K, T, r, q, v0, kappa, theta, sigma, rho, lambd, mu_j, delta_j):
        """
        Approximate Bates model price using volatility adjustment
        This is a practical approximation for calibration purposes
        """
        # Total variance including jump component
        jump_variance = lambd * (mu_j**2 + delta_j**2)
        total_variance = theta + jump_variance
        
        # Adjust for mean jump size effect
        jump_adjustment = lambd * (np.exp(mu_j + 0.5 * delta_j**2) - 1)
        
        # Effective parameters for Black-Scholes approximation
        effective_vol = np.sqrt(total_variance)
        effective_r = r - jump_adjustment
        
        # Black-Scholes with adjusted parameters
        d1 = (np.log(S0/K) + (effective_r - q + 0.5 * effective_vol**2) * T) / (effective_vol * np.sqrt(T))
        d2 = d1 - effective_vol * np.sqrt(T)
        
        call_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-effective_r * T) * norm.cdf(d2)
        return max(call_price, 0.01)  # Ensure positive price
    
    @staticmethod
    def black_scholes(S0, K, T, r, q, sigma):
        """Standard Black-Scholes formula"""
        d1 = (np.log(S0/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

def bates_model_objective(params, market_prices, strikes, days_to_exp, S0, r, q):
    """
    Objective function for Bates model calibration
    """
    v0, kappa, theta, sigma, rho, lambd, mu_j, delta_j = params
    
    total_error = 0.0
    valid_points = 0
    
    for i, (market_price, K, T) in enumerate(zip(market_prices, strikes, days_to_exp)):
        if market_price > 0.01:  # Only use liquid options
            # Calculate Bates model price
            model_price = BatesPricer.bates_approximation(S0, K, T, r, q, v0, kappa, theta, sigma, rho, lambd, mu_j, delta_j)
            
            # Relative squared error (better for options with different prices)
            error = ((market_price - model_price) / market_price) ** 2
            total_error += error
            valid_points += 1
    
    # Add regularization to prevent extreme parameters
    regularization = 0.01 * (
        (kappa - 2.0)**2 + (theta - 0.04)**2 + (lambd - 1.0)**2 + 
        (mu_j + 0.05)**2 + (rho + 0.7)**2
    )
    
    return total_error / valid_points + regularization if valid_points > 0 else 1e6

def calibrate_bates_from_chain(option_chain, S0, r, q):
    """
    Calibrate Bates parameters from an option chain
    """
    print("Starting Bates model calibration...")
    
    # Extract data from option chain
    strikes = option_chain['strike'].values
    market_prices = option_chain['mid_price'].values
    days_to_exp = option_chain['days_to_exp'].values
    
    # Initial parameter guesses (sensible defaults)
    initial_guess = [
        0.04,    # v0: initial variance (20% vol)
        2.0,     # kappa: mean-reversion speed
        0.04,    # theta: long-term variance (20% vol)
        0.3,     # sigma: vol of vol
        -0.7,    # rho: correlation
        1.0,     # lambda: jump intensity
        -0.05,   # mu_j: mean jump size
        0.10     # delta_j: jump volatility
    ]
    
    # Parameter bounds (to ensure reasonable values)
    bounds = [
        (0.01, 0.5),    # v0
        (0.1, 10.0),    # kappa
        (0.01, 0.5),    # theta
        (0.1, 1.0),     # sigma
        (-0.99, 0.0),   # rho (typically negative)
        (0.1, 20.0),    # lambda
        (-0.3, 0.3),    # mu_j (can be positive or negative)
        (0.05, 0.5)     # delta_j
    ]
    
    # Perform optimization
    result = minimize(
        bates_model_objective,
        initial_guess,
        args=(market_prices, strikes, days_to_exp, S0, r, q),
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 100, 'disp': True}
    )
    
    return result

def create_sample_option_chain(S0=100, r=0.05, q=0.02, T=0.1):
    """
    Create a realistic sample option chain for testing
    """
    # Create strikes around current price
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    
    # True Bates parameters to generate "market" prices
    true_params = [0.04, 2.0, 0.04, 0.3, -0.7, 2.0, -0.08, 0.15]
    v0, kappa, theta, sigma, rho, lambd, mu_j, delta_j = true_params
    
    # Generate market prices with some noise
    market_prices = []
    for K in strikes:
        true_price = BatesPricer.bates_approximation(S0, K, T, r, q, v0, kappa, theta, sigma, rho, lambd, mu_j, delta_j)
        # Add small random noise to simulate real market
        noisy_price = true_price * np.random.uniform(0.98, 1.02)
        market_prices.append(max(noisy_price, 0.01))
    
    option_chain = pd.DataFrame({
        'strike': strikes,
        'bid': [p * 0.99 for p in market_prices],  # Simulate bid-ask spread
        'ask': [p * 1.01 for p in market_prices],
        'days_to_exp': [T] * len(strikes)
    })
    option_chain['mid_price'] = (option_chain['bid'] + option_chain['ask']) / 2
    
    return option_chain, true_params

def plot_calibration_results(option_chain, S0, r, q, calibrated_params, true_params=None):
    """Plot market prices vs calibrated model prices"""
    strikes = option_chain['strike'].values
    market_prices = option_chain['mid_price'].values
    T = option_chain['days_to_exp'].iloc[0]
    
    # Calculate model prices with calibrated parameters
    model_prices = []
    for K in strikes:
        price = BatesPricer.bates_approximation(S0, K, T, r, q, *calibrated_params)
        model_prices.append(price)
    
    # Calculate true prices if available
    true_prices = []
    if true_params is not None:
        for K in strikes:
            price = BatesPricer.bates_approximation(S0, K, T, r, q, *true_params)
            true_prices.append(price)
    
    plt.figure(figsize=(12, 8))
    
    # Plot prices
    plt.subplot(2, 1, 1)
    plt.plot(strikes, market_prices, 'bo-', label='Market Prices', linewidth=2, markersize=6)
    plt.plot(strikes, model_prices, 'ro--', label='Calibrated Bates Model', linewidth=2, markersize=6)
    
    if true_params is not None:
        plt.plot(strikes, true_prices, 'g--', label='True Bates Model', linewidth=2, alpha=0.7)
    
    plt.axvline(S0, color='k', linestyle='--', alpha=0.5, label='Spot Price')
    plt.xlabel('Strike Price')
    plt.ylabel('Option Price')
    plt.title('Bates Model Calibration Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot errors
    plt.subplot(2, 1, 2)
    errors = (np.array(model_prices) - market_prices) / market_prices * 100
    plt.bar(strikes, errors, alpha=0.7, color='orange')
    plt.axhline(0, color='k', linestyle='-')
    plt.xlabel('Strike Price')
    plt.ylabel('Pricing Error (%)')
    plt.title('Calibration Errors')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def example_calibration():
    """Complete working example of Bates model calibration"""
    print("=== BATES MODEL CALIBRATION EXAMPLE ===")
    
    # Parameters
    S0 = 100.0
    r = 0.05
    q = 0.02
    
    # Create sample option chain
    option_chain, true_params = create_sample_option_chain(S0, r, q)
    
    print("Sample Option Chain:")
    print(option_chain[['strike', 'mid_price']].to_string(index=False))
    
    # Perform calibration
    result = calibrate_bates_from_chain(option_chain, S0, r, q)
    
    if result.success:
        calibrated_params = result.x
        param_names = ['v0', 'kappa', 'theta', 'sigma', 'rho', 'lambda', 'mu_j', 'delta_j']
        
        print("\n" + "="*50)
        print("CALIBRATION RESULTS:")
        print("="*50)
        
        # Print comparison if we know true parameters
        if true_params is not None:
            print(f"{'Parameter':<10} {'True':<8} {'Calibrated':<12} {'Error':<8}")
            print("-" * 45)
            for name, true, calib in zip(param_names, true_params, calibrated_params):
                error_pct = (calib - true) / true * 100
                print(f"{name:<10} {true:<8.4f} {calib:<12.4f} {error_pct:<8.1f}%")
        else:
            for name, value in zip(param_names, calibrated_params):
                print(f"{name:<10}: {value:.4f}")
        
        # Plot results
        plot_calibration_results(option_chain, S0, r, q, calibrated_params, true_params)
        
        # Economic interpretation
        print("\n" + "="*50)
        print("ECONOMIC INTERPRETATION:")
        print("="*50)
        v0, kappa, theta, sigma, rho, lambd, mu_j, delta_j = calibrated_params
        print(f"• Initial Volatility: {np.sqrt(v0)*100:.1f}%")
        print(f"• Long-run Volatility: {np.sqrt(theta)*100:.1f}%")  
        print(f"• Vol Mean-reversion Speed: {kappa:.2f} (1/kappa = {1/kappa:.2f} years)")
        print(f"• Vol-of-Vol: {sigma*100:.1f}%")
        print(f"• Leverage Effect (ρ): {rho:.2f}")
        print(f"• Jump Intensity: {lambd:.2f} jumps/year")
        print(f"• Average Jump Size: {mu_j*100:.1f}%")
        print(f"• Jump Uncertainty: {delta_j*100:.1f}%")
        
        return calibrated_params
    else:
        print("Calibration failed:", result.message)
        return None

# Run the example
example_calibration()



#| include: false 
#| eval: false
#| cache: true
option_chain = options_df
event_date = pd.to_datetime('2025-12-15')
option_chain['mid_price'] = (option_chain['bid'] + option_chain['ask']) / 2

S0 = 100.0  # Current underlying price (get from market data)
r = 0.05    # Risk-free rate (get from Treasury yields)
q = 0.02    # Dividend yield (get from stock data)

def analyze_event_effect(event_date, days_before=30, days_after=30):
    """
    Analyze how an event affects Bates parameters
    """
    # Pre-event calibration (using options from before the event)
    pre_chain, S0_pre, r_pre, q_pre = option_chain[option_chain['expiration'] - (event_date)<=pd.Timedelta(0)], S0, r, q
    pre_params = calibrate_bates_from_chain(pre_chain, S0_pre, r_pre, q_pre)
    
    # Post-event calibration  
    post_chain, S0_post, r_post, q_post = option_chain[option_chain['expiration'] - (event_date)>pd.Timedelta(0)], S0+10, r, q
    post_params = calibrate_bates_from_chain(post_chain, S0_post, r_post, q_post)
    
    # Compare parameter changes
    param_names = ['v0', 'kappa', 'theta', 'sigma', 'rho', 'lambda', 'mu_j', 'delta_j']
    
    print("Parameter Changes due to Event:")
    for name, pre, post in zip(param_names, pre_params.x, post_params.x):
        change_pct = (post - pre) / pre * 100
        print(f"{name:8}: {pre:.4f} → {post:.4f} ({change_pct:+.1f}%)")
    
    return pre_params, post_params

analyze_event_effect(event_date = event_date)