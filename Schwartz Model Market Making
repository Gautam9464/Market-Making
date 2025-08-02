import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Step 1: Download historical data
def get_data(symbol="AAPL", period="2y"):
    df = yf.download(symbol, period=period)
    df = df["Close"].dropna()
    return df


# Step 2: Estimate parameters for Schwartz model
def estimate_schwartz_params(prices, dt=1):
    log_prices = np.log(prices)
    delta_log = log_prices.diff().dropna()

    Y = delta_log[1:]
    X = -1 * (log_prices.shift(1).dropna()[1:] - log_prices.mean())

    gamma = np.linalg.lstsq(X.values.reshape(-1, 1), Y.values, rcond=None)[0][0]
    residuals = Y - gamma * X
    sigma = np.std(residuals) / np.sqrt(dt)
    mu = np.exp(log_prices.mean())

    return gamma, sigma, mu


# Step 3: Simulate price under Schwartz model (lognormal OU)
class SchwartzSimulator:
    def _init_(self, mu, gamma, sigma, Q0, T=1000, dt=1):
        self.mu = mu
        self.gamma = gamma
        self.sigma = sigma
        self.Q0 = Q0
        self.T = T
        self.dt = dt
        self.steps = int(T / dt)

    def simulate(self):
        logQ = np.log(self.Q0)
        prices = []

        for _ in range(self.steps):
            dW = np.random.normal(0, np.sqrt(self.dt))
            drift = -self.gamma * (logQ - np.log(self.mu))
            logQ += drift * self.dt + self.sigma * dW
            prices.append(np.exp(logQ))

        return np.array(prices)


# Step 4: Market Making Strategy
class MarketMaker:
    def _init_(self, spread=1.0, inventory_limit=100):
        self.inventory = 0
        self.pnl = 0
        self.spread = spread
        self.inventory_limit = inventory_limit
        self.pnl_series = []
        self.inventory_series = []

    def quote(self, price):
        inventory_skew = (self.inventory / self.inventory_limit) * self.spread
        bid = price - self.spread / 2 - inventory_skew
        ask = price + self.spread / 2 - inventory_skew
        return bid, ask

    def run(self, prices):
        for price in prices:
            bid, ask = self.quote(price)

            r = np.random.rand()
            if r < 0.5:
                self.inventory += 1
                self.pnl -= bid
            else:
                self.inventory -= 1
                self.pnl += ask

            self.pnl_series.append(self.pnl)
            self.inventory_series.append(self.inventory)

        self.pnl += self.inventory * prices[-1]
        return self.pnl


# Run everything
symbol = "INFY.NS"
real_prices = get_data(symbol)
gamma, sigma, mu = estimate_schwartz_params(real_prices)


simulator = SchwartzSimulator(mu=mu, gamma=gamma, sigma=sigma, Q0=real_prices.iloc[-1], T=1000)
simulated_prices = simulator.simulate()

mm = MarketMaker(spread=1.0)
final_pnl = mm.run(simulated_prices)

# Plot
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(simulated_prices)
plt.title(f'Schwartz Model Simulation Based on {symbol} Parameters')

plt.subplot(2, 1, 2)
plt.plot(mm.pnl_series, label='PnL')
plt.plot(mm.inventory_series, label='Inventory')
plt.legend()
plt.title('Market Making Results')
plt.tight_layout()
plt.show()
