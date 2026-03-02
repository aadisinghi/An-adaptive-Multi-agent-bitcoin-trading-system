""" 
Input: Portfolio history, recent returns

"""

""" 
A. Portfolio Daily Returns

btc_daily = btc_minute['Close'].resample('1D').last().pct_change().dropna()
portfolio_returns = btc_alloc * btc_daily  # cash is risk-free

B. VaR and CVaR (95%, 1-day)

lookback = 90
window_returns = portfolio_returns[-lookback:]
var = -np.percentile(window_returns, 5)
cvar = -window_returns[window_returns <= -var].mean()

C. Realized Volatility (7-day)

btc_minute['log_return'] = np.log(btc_minute['Close'] / btc_minute['Close'].shift(1))
rv_daily = btc_minute['log_return'].pow(2).resample('1D').sum().pow(0.5)
rv_7d = rv_daily.rolling(7).mean().iloc[-1]

D. 7-day Rolling Return

rolling_7d_return = portfolio_returns[-7:].sum()

max_loss = investment_amount * risk_tolerance  # e.g. $250
rv_threshold = 0.10  # ~10% realized vol over 7d triggers concern

if cvar > max_loss:
    decision = "REJECT"
    reason = f"CVaR = ${cvar:.2f} exceeds user loss limit ${max_loss:.2f}"
    
elif var > max_loss or rv_7d > rv_threshold:
    if rolling_7d_return < 0:
        decision = "REJECT"
        reason = (
            f"VaR or RV above threshold AND 7-day return is negative. "
            f"RV = {rv_7d:.2f}, VaR = ${var:.2f}, Return = {rolling_7d_return:.2%}"
        )
    else:
        decision = "WARNING"
        reason = (
            f"VaR = ${var:.2f} or RV = {rv_7d:.2f} approaching limits, but performance stable"
        )
else:
    decision = "ACCEPT"
    reason = (
        f"Risk metrics within bounds. VaR = ${var:.2f}, CVaR = ${cvar:.2f}, RV = {rv_7d:.2f}"
    )

    
Phase	Data Used	Behavior	Memory Impact
Training	July → T	Daily risk evaluation → feedback to Reflect → Decision reallocates	Reflect logs risk interactions
Testing	July → T	Same logic, but Decision doesn’t learn — only executes	Risk logs warnings but doesn’t retrain agents

"""