This project has provided us with a comprehensive, hands-on exploration of market making strategies, beginning with a simple bid-ask spread model and evolving into a sophisticated, modular simulation framework. Throughout our journey, we incorporated key financial concepts such as inventory management, volatility-aware quoting, order book dynamics, and stochastic trade execution. Each iteration of the strategy progressively brought us closer to replicating real-world market behavior while maintaining the flexibility for experimentation and learning.

FEATURES:-
Order book simulation with configurable depth

1. Market making with risk-managed inventory constraints
2. Historical price data via yfinance for all stocks
3. Real-time inventory, cash, and PnL tracking
4. Performance metrics table including:
     Sharpe Ratio
     Max Drawdown (%)
     Turnover (Number of Trades)
     Margin Usage (%)
     Total PnL
     Net PnL
5. Visualization of cumulative PnL over time for selected stock.
6. Interactive GUI table to browse stock-wise performance (via Tkinter)

We ran the code for top 50 Indian as well as USA stocks and observed decent performance in both cases.

Our simulations, spanning both Indian and US equities, revealed important insights into how different markets respond to liquidity provisioning. While early strategies showed promise, subsequent enhancements—like the use of high-frequency data, stop-loss controls, and synthetic market impact—significantly improved performance stability and risk-adjusted returns.

Moreover, the GUI-based visualization and portfolio-level metrics allowed for intuitive evaluation of results and strategy robustness. We also identified exciting avenues for future development, such as integrating the Schwartz model for synthetic price generation and using true Level 2 data to achieve institutional-grade realism.

This project not only deepened our understanding of market microstructure and algorithmic trading but also sharpened our skills in simulation design, risk modeling, and data analysis—making it a truly rewarding SoC experience.
