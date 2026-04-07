# Backtest Results

![Equity Curve](results/1_equity_curve.png)
![Buy & Hold Comparison](results/2_vs_buyhold.png)

# Inspiration
Strategy components adapted from:

https://github.com/neurotrader888/VolatilityHawkes

https://github.com/neurotrader888/IntramarketDifference

These examples use hourly historical BTC (Bitcoin) and ETH (Ethereum) data. Small changes and adjustments make it fully compatible with the historical data the strategy uses, namely, resampled 1 minute OHLCV for NQ, ES, and YM.

# Overview
This strategy only trades NQ and it uses ES and YM for relative strength comparison. To quantify this, first, we use a CMMA (Close Minus Moving Average) which can be defined for each asset $i\in \{ \text{NQ}, \text{ES}, \text{YM} \}$ with a lookback window $L$,
```math
\text{CMMA}^{(i)}_t=\frac{\text{Close}^{(i)}_t-\text{MA}^{(i)}_t}{\text{ATR}^{(i)}_t\cdot\sqrt{L}}
```
with 
```math
\text{MA}_t=\frac{1}{L}\sum_{i=0}^{L-1}\text{Close}_{t-i}\qquad\text{(Rolling Mean / SMA)}
```
and
```math
\text{ATR}_t=\frac{1}{L}\text{TR}_t+\left(1-\frac{1}{L}\right)\text{ATR}_{t-1} \qquad \text{(Wilder's ATR)}
```
where
```math
\text{TR}_t=\max(\text{High}_t-\text{Low}_{t}, |\text{High}_t-\text{Close}_{t-1}|, |\text{Low}_t-\text{Close}_{t-1}|)
```
```math
\text{ATR}_0:=\frac{1}{L}\sum_{i=1}^{L}\text{TR}_{i} \qquad \text{(Initial ATR value)}
```

After computing the CMMA on each asset, we build features that capture relative strength and divergence across indices. Instead of using a single CMMA, the strategy builds a composite CMMA across multiple lookbacks:

```math
\text{Composite}^{(i)}_t =
w_{\text{short}} \cdot \text{CMMA}^{(i)}_{\text{short},t}
+ w_{\text{mid}} \cdot \text{CMMA}^{(i)}_{\text{mid},t}
+ w_{\text{long}} \cdot \text{CMMA}^{(i)}_{\text{long},t}
```

which allows the model to capture multi-timescale deviations rather than relying on a single horizon.

---

# Feature Engineering

Using the composite values, we define relative strength features:

$$
S^{(\text{NQ-ES})}_t = \text{Composite}^{(\text{NQ})}_t - \text{Composite}^{(\text{ES})}_t
$$

$$
S^{(\text{NQ-YM})}_t = \text{Composite}^{(\text{NQ})}_t - \text{Composite}^{(\text{YM})}_t
$$

We also define a unified **advantage signal**:

$$
\text{Advantage}_t = \text{Composite}^{(\text{NQ})}_t - \frac{1}{2}\left(\text{Composite}^{(\text{ES})}_t + \text{Composite}^{(\text{YM})}_t\right)
$$

This represents how strongly NQ is outperforming or underperforming relative to ES and YM.

---

# Hawkes Process

To filter trades by market regime, the strategy applies a self-exciting Hawkes-style process to volatility:

$$
H_t = \kappa \sum_{i=0}^{t} e^{-\kappa (t-i)} \cdot \frac{\text{High}_i - \text{Low}_i}{\text{ATR}_i}
$$

This gives a measure of volatility clustering.

Rolling quantiles are then used to define regimes, for example,

- Low quantile → quiet/reset state  
- High quantile → breakout regime  

A directional regime is determined by comparing price to a quiet-period anchor:

- Price above anchor → bullish regime  
- Price below anchor → bearish regime  

---

### Long Conditions

A long position is entered when:

- NQ has strong positive advantage  
- NQ is leading ES and YM  

$$
\text{Advantage}_t > \theta_{\text{entry}}
$$

---

### Short Conditions

A short position is entered when:

- NQ has strong negative advantage  
- NQ is lagging ES and YM  

$$
\text{Advantage}_t < -\theta_{\text{entry}}
$$

---

# Exit Logic

There are no TPs in this model as it relies on the strategy hypothesis being invalidated live in order to have a clear exit. 

Some of the conditions that have to be satisfied for an exit:
- Advantage weakens or reverses
- Regime resets for a sustained period
- Maximum hold 
- SL based on ATR multiple  

---

# Risk Management

Position sizing is volatility-adjusted:

$$
\text{Contracts} =
\frac{\text{Target Risk}}{\text{ATR} \cdot \text{SL Multiplier} \cdot \text{Point Value}}
$$

(clamped to a maximum number of contracts)
