import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import tkinter as tk
from tkinter import ttk

# ----------------------- Order Book Simulation -----------------------
class OrderBook:
    def __init__(self, center_price, levels=5, tick_size=1, size_per_level=10):
        self.center_price = float(center_price.item()) if isinstance(center_price, np.ndarray) else float(center_price[0]) if isinstance(center_price, list) else float(center_price)
        self.levels = levels
        self.tick_size = tick_size
        self.size_per_level = size_per_level
        self.update_book()

    def update_book(self):
        self.bids = {self.center_price - i * self.tick_size: self.size_per_level for i in range(1, self.levels + 1)}
        self.asks = {self.center_price + i * self.tick_size: self.size_per_level for i in range(1, self.levels + 1)}

    def get_top_bid(self):
        return max(self.bids.keys())

    def get_top_ask(self):
        return min(self.asks.keys())

    def simulate_trade(self, side, price, size):
        if side == "buy":
            for lvl_price in sorted(self.asks.keys()):
                if price >= lvl_price and size > 0:
                    trade_size = min(size, self.asks[lvl_price])
                    self.asks[lvl_price] -= trade_size
                    size -= trade_size
                    if self.asks[lvl_price] <= 0:
                        del self.asks[lvl_price]
        elif side == "sell":
            for lvl_price in sorted(self.bids.keys(), reverse=True):
                if price <= lvl_price and size > 0:
                    trade_size = min(size, self.bids[lvl_price])
                    self.bids[lvl_price] -= trade_size
                    size -= trade_size
                    if self.bids[lvl_price] <= 0:
                        del self.bids[lvl_price]

# --------------------- Market Maker Class ---------------------
class MarketMaker:
    def __init__(self, initial_cash, base_spread=2.0, max_inventory=20):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.inventory = 0
        self.spread = base_spread
        self.max_inventory = max_inventory
        self.trade_history = []
        self.quotes = []
        self.pnl_history = []
        self.inventory_history = []
        self.cash_history = []

    def generate_quotes(self, fair_value):
        fair_value = float(fair_value.item()) if isinstance(fair_value, np.ndarray) else float(fair_value[0]) if isinstance(fair_value, list) else float(fair_value)
        inventory_skew = (self.inventory / self.max_inventory) * self.spread
        bid = fair_value - self.spread - inventory_skew
        ask = fair_value + self.spread - inventory_skew
        self.quotes.append((round(bid, 2), round(ask, 2)))
        return round(bid, 2), round(ask, 2)

    def position_size(self):
        return 1

    def execute_trade(self, side, price, size):
        if abs(self.inventory) > self.max_inventory * 0.8:
            if (side == "buy" and self.inventory < 0) or (side == "sell" and self.inventory > 0):
                pass
            else:
                return

        if side == "buy":
            cost = price * size
            if self.cash >= cost and abs(self.inventory + size) <= self.max_inventory:
                self.cash -= cost
                self.inventory += size
                self.trade_history.append(("BUY", price, size))
        elif side == "sell":
            if self.inventory >= size:
                self.cash += price * size
                self.inventory -= size
                self.trade_history.append(("SELL", price, size))

    def record_state(self, fair_value):
        self.cash_history.append(self.cash)
        self.inventory_history.append(self.inventory)
        fair_value = float(fair_value.item()) if isinstance(fair_value, np.ndarray) else float(fair_value[0]) if isinstance(fair_value, list) else float(fair_value)
        realized = self.cash - self.initial_cash
        unrealized = self.inventory * fair_value
        self.pnl_history.append(realized + unrealized)

# ------------------- Simulation Loop -------------------
def run_simulation(fair_values):
    mm = MarketMaker(initial_cash=100000, base_spread=1.0)
    for fair_value in fair_values.tolist():
        ob = OrderBook(center_price=fair_value)
        bid, ask = mm.generate_quotes(fair_value)
        size = mm.position_size()

        if ask <= ob.get_top_ask():
            mm.execute_trade("sell", ask, size)
            ob.simulate_trade("sell", ask, size)

        if bid >= ob.get_top_bid():
            mm.execute_trade("buy", bid, size)
            ob.simulate_trade("buy", bid, size)

        mm.record_state(fair_value)

    mm.pnl_history = [v - mm.pnl_history[0] for v in mm.pnl_history]
    return mm

# -------------------- Execution --------------------
nifty50_symbols = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "HCLTECH.NS",
    "ASIANPAINT.NS", "LT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS", "TECHM.NS", "HDFCLIFE.NS", "BAJAJFINSV.NS",
    "POWERGRID.NS", "NTPC.NS", "JSWSTEEL.NS", "TATAMOTORS.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "GRASIM.NS", "TATASTEEL.NS", "ONGC.NS", "BPCL.NS", "EICHERMOT.NS",
    "BRITANNIA.NS", "SHREECEM.NS", "CIPLA.NS", "DIVISLAB.NS", "HINDALCO.NS", "BAJAJ-AUTO.NS",
    "INDUSINDBK.NS", "HDFCAMC.NS", "SBILIFE.NS", "UPL.NS", "DRREDDY.NS", "M&M.NS",
    "HEROMOTOCO.NS", "APOLLOHOSP.NS"
]

data = yf.download(nifty50_symbols, period="3mo", interval="1d")["Close"].dropna(axis=1, how="any")

performance = {}
example_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

plt.figure(figsize=(12, 6))
for symbol in data.columns:
    mm = run_simulation(data[symbol].dropna().values)
    returns = np.diff(mm.pnl_history)
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    max_drawdown = np.max(np.maximum.accumulate(mm.pnl_history) - mm.pnl_history)
    drawdown_pct = max_drawdown / np.maximum.accumulate(mm.pnl_history).max() * 100 if np.maximum.accumulate(mm.pnl_history).max() > 0 else 0
    turnover = len(mm.trade_history)
    avg_inventory = np.mean(np.abs(mm.inventory_history))
    margin_pct = avg_inventory / mm.max_inventory * 100
    net_pnl = mm.cash + mm.inventory * data[symbol].iloc[-1] - mm.initial_cash
    performance[symbol] = [symbol, sharpe_ratio, drawdown_pct, turnover, margin_pct, mm.pnl_history[-1], net_pnl]

    if symbol in example_stocks and len(mm.pnl_history) > 0:
        plt.plot(mm.pnl_history, label=symbol)

plt.title("Cumulative PnL Over Time (Selected Stocks)")
plt.xlabel("Time Step")
plt.ylabel("PnL")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

performance_df = pd.DataFrame.from_dict(performance, orient='index', columns=[
    'Stock', 'Sharpe Ratio', 'Max Drawdown (%)', 'Turnover (%)', 'Margin Usage (%)', 'Total PnL', 'Net PnL'
])

performance_df.to_csv("nifty50_simulation_results.csv")

performance_df[['Total PnL']].sort_values('Total PnL').plot(kind='barh', figsize=(10, 12), title='PnL across NIFTY 50 Stocks')
plt.tight_layout()
plt.grid()
plt.show()

# GUI Table Output
root = tk.Tk()
root.title("Market Making Performance Metrics")

frame = ttk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

cols = performance_df.columns.tolist()
tree = ttk.Treeview(frame, columns=cols, show='headings')
for col in cols:
    tree.heading(col, text=col)
    tree.column(col, anchor=tk.CENTER)

tree.pack(fill=tk.BOTH, expand=True)

for _, row in performance_df.iterrows():
    tree.insert("", tk.END, values=[f"{v:.2f}" if isinstance(v, float) else v for v in row.values])

root.mainloop()
