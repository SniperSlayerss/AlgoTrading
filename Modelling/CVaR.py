import bt
import pandas
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

data = bt.get('aapl, nvda, msft', start='2015-01-01')
weights = np.array([0.4, 0.5, 0.1])
invested = 100000

# Calculate percentage change (daily return)
percent_change = data.pct_change().dropna()

# Calculate portfolio returns for the last 'lookback' days
lookback = 100
portfolio_returns = percent_change.iloc[-lookback:].dot(weights)

def calculate_var(alpha, invested, portfolio_returns):
   std = portfolio_returns.std()
   mean = portfolio_returns.mean()
   z_score = norm.ppf(1 - alpha)
   return -(mean + z_score * std) * invested

def calculate_cvar(alpha, invested, portfolio_returns):
    # mean of portfolio (weighted stock values averaged)
    var_threshold = portfolio_returns.quantile(alpha)
    expected_shortfall = portfolio_returns[portfolio_returns < var_threshold].mean()
    return -invested * expected_shortfall

alpha = 0.05
CVaR = calculate_cvar(alpha, invested, portfolio_returns)
VaR = calculate_var(alpha, invested, portfolio_returns)

VaR_return = VaR / invested
CVaR_return = CVaR / invested

print(f"CVaR: {CVaR}, VaR: {VaR}")
