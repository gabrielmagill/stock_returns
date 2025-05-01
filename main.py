import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def simulate_stock_price(S0, years, mu, sigma):
    N = years * 252  # 252 trading days per year
    T = float(years)
    dt = T/N
    
    prices = np.zeros(N+1)
    prices[0] = S0
    
    rand_increments = np.random.normal(0, 1, N)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand_increments
    prices[1:] = S0 * np.exp(np.cumsum(log_returns))
    
    # Generate business days
    start_date = datetime(2001, 1, 1)
    dates = pd.date_range(start=start_date, periods=N+1, freq='B')
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    
    return df

def calculate_return_percentiles(df, periods_years, dividend_rate=0.01):
    results = []
    
    for period_years in periods_years:
        period_days = round(period_years * 252)
        returns = []
        
        for start_idx in range(0, len(df) - period_days, 5):
            start_price = df['Price'].iloc[start_idx]
            end_price = df['Price'].iloc[start_idx + period_days]
            
            # Add dividend accumulation factor e^(d*T)
            dividend_factor = np.exp(dividend_rate * period_years)
            # Total return includes both price appreciation and dividends
            return_over_period = (end_price/start_price) * dividend_factor - 1
            returns.append(return_over_period)
            
        results.append([
            period_years,
            (np.array(returns) < 0).mean(),
            np.percentile(returns, 99),
            np.percentile(returns, 1),
            np.mean(returns),
            np.std(returns),
        ])

    df_results = pd.DataFrame(results, columns=['period_years', 'probability_negative_returns', 'p99', 'p1', 'mean', 'std'])
    return df_results

if __name__ == "__main__":
    # Generate stock prices
    S0 = 100.0
    years = 30
    mu = 0.15
    sigma = 0.30
    df = simulate_stock_price(S0, years, mu, sigma)
    
    # Calculate return percentiles
    df_results = calculate_return_percentiles(df, periods_years=np.linspace(0.01, 15, 100))
    
    # Plot percentiles
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    ax.fill_between(df_results['period_years'], df_results['p1'], df_results['p99'], alpha=0.2, color='gray', label='1st-99th percentile range')
    ax2.plot(df_results['period_years'], df_results['probability_negative_returns'], 'b-', label='Probability of Negative Returns')
    ax2.set_ylabel('Probability of Negative Returns')
    ax2.set_xlabel('Investment Horizon (years)')
    
    ax.set_ylabel('Return')
    ax.set_xlabel('Investment Horizon (years)')
    ax.set_title('Return Percentiles by Investment Horizon')
    ax.legend()   
    
    fig.tight_layout()
    fig.savefig('return_percentiles.png')
    plt.close()
    

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df['Price'], 'b-', label='Stock Price')
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    param_text = f'Parameters:\nInitial Price: ${S0:.2f}\nDrift (μ): {mu:.2f}\nVolatility (σ): {sigma:.2f}\nTime Horizon: {years} years'
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    ax.set_title('Simulated Stock Price Over Time')
    ax.grid(True)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig('stock_price.png')
    plt.close()
