
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import time as _time

# config

@dataclass
class Config:
    data_dir: str = "data"
    files: dict = field(default_factory=lambda: {
        "YM": "ym1m_continuous.parquet",
        "ES": "es1m_continuous.parquet",
        "NQ": "nq1m_continuous.parquet",
    })
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    trade_asset: str = "NQ"

    # nq contract specs (fees on ninjatrader)
    point_value: float = 20.0
    fee_per_side: float = 1.40
    tick_size: float = 0.25
    slippage_ticks: float = 0.0

    # CMMA
    cmma_short: int = 20
    cmma_mid: int = 60
    cmma_long: int = 180
    atr_period: int = 14
    w_short: float = 0.25
    w_mid: float = 0.50
    w_long: float = 0.25

    # hawkes thresholds
    hawkes_kappa: float = 0.02
    hawkes_quantile_window: int = 500
    hawkes_q_low: float = 0.05
    hawkes_q_high: float = 0.95

    # signal thresholds
    entry_spread_threshold: float = 0.3
    exit_spread_threshold: float = -0.05
    min_hold_bars: int = 10
    max_hold_bars: int = 150
    stop_loss_atr_mult: float = 4.0
    take_profit_atr_mult: float = 6.0
    hawkes_exit_cooldown: int = 3

    # rth filter (UTC times: 13:30 = 9:30 ET, 20:00 = 16:00 ET)
    rth_only: bool = True
    rth_start_hour: int = 13
    rth_start_min: int = 30
    rth_end_hour: int = 20
    rth_end_min: int = 0

    # position sizing
    target_risk_dollars: float = 500.0
    max_contracts: int = 5
    use_dynamic_sizing: bool = True

    #"RS_HAWKES" (default) or "RS_ONLY"
    mode: str = "RS_HAWKES"

    resample_freq: Optional[str] = "5min"
    results_dir: str = "results"


CFG = Config()

# data stuff

def load_data(cfg: Config) -> dict[str, pd.DataFrame]:
    data = {}
    for sym, fname in cfg.files.items():
        df = pd.read_parquet(Path(cfg.data_dir) / fname)
        df.index.name = "timestamp"
        df = df[["open", "high", "low", "close", "volume"]].copy().sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if cfg.start_date:
            df = df.loc[cfg.start_date:]
        if cfg.end_date:
            df = df.loc[:cfg.end_date]
        data[sym] = df
    return data


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    return df.resample(freq).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["open"])


def align_data(data: dict[str, pd.DataFrame], freq: Optional[str] = None) -> dict[str, pd.DataFrame]:
    if freq:
        data = {sym: resample_ohlcv(df, freq) for sym, df in data.items()}
    idx = data["YM"].index
    for sym in data:
        idx = idx.intersection(data[sym].index)
    idx = idx.sort_values()
    return {sym: df.loc[idx].copy() for sym, df in data.items()}


# features

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_cmma(close: pd.Series, atr: pd.Series, lookback: int) -> pd.Series:
    ma = close.rolling(lookback, min_periods=lookback).mean()
    return (close - ma) / (atr * np.sqrt(lookback))


def compute_hawkes(df: pd.DataFrame, atr: pd.Series, kappa: float) -> pd.Series:
    """Recursive self-exciting filter on volatility-normalized range."""
    shock = ((df["high"] - df["low"]) / atr).fillna(0.0).values
    alpha = np.exp(-kappa)
    n = len(shock)
    hawkes = np.empty(n, dtype=np.float64)
    hawkes[0] = shock[0]
    for i in range(1, n):
        hawkes[i] = hawkes[i - 1] * alpha + shock[i]
    return pd.Series(hawkes * kappa, index=df.index)


def engineer_features(data: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    syms = list(data.keys())
    feat = pd.DataFrame(index=data[syms[0]].index)

    composites = {}
    for sym in syms:
        df = data[sym]
        atr = compute_atr(df, cfg.atr_period)
        composite = (cfg.w_short * compute_cmma(df["close"], atr, cfg.cmma_short)
                     + cfg.w_mid * compute_cmma(df["close"], atr, cfg.cmma_mid)
                     + cfg.w_long * compute_cmma(df["close"], atr, cfg.cmma_long))

        composites[sym] = composite
        feat[f"{sym}_composite"] = composite
        feat[f"{sym}_close"] = df["close"]
        feat[f"{sym}_atr"] = atr
        feat[f"{sym}_hawkes"] = compute_hawkes(df, atr, cfg.hawkes_kappa)
        feat[f"{sym}_open"] = df["open"]

    # nq relative features
    feat["nq_es_spread"] = composites["NQ"] - composites["ES"]
    feat["nq_ym_spread"] = composites["NQ"] - composites["YM"]
    feat["nq_advantage"] = composites["NQ"] - (composites["ES"] + composites["YM"]) / 2.0

    comp_cols = [f"{s}_composite" for s in syms]
    feat["nq_rank"] = feat[comp_cols].rank(axis=1, ascending=False)["NQ_composite"]
    feat["max_spread"] = feat[comp_cols].values.max(axis=1) - feat[comp_cols].values.min(axis=1)

    # hawkes quantile bands
    qw = cfg.hawkes_quantile_window
    for sym in syms:
        h = feat[f"{sym}_hawkes"]
        feat[f"{sym}_hawkes_q05"] = h.rolling(qw, min_periods=qw).quantile(cfg.hawkes_q_low)
        feat[f"{sym}_hawkes_q95"] = h.rolling(qw, min_periods=qw).quantile(cfg.hawkes_q_high)

    # rth flag
    time_minutes = feat.index.hour * 60 + feat.index.minute
    rth_start = cfg.rth_start_hour * 60 + cfg.rth_start_min
    rth_end = cfg.rth_end_hour * 60 + cfg.rth_end_min
    feat["is_rth"] = (time_minutes >= rth_start) & (time_minutes < rth_end)

    return feat


# hawkes regime detector

def compute_hawkes_regimes(feat: pd.DataFrame, syms: list[str]) -> pd.DataFrame:
    """Track quiet anchors and breakout crossings per asset.
    regime: +1 bullish, -1 bearish, 0 neutral (quiet/reset)."""
    n = len(feat)
    for sym in syms:
        hawkes = feat[f"{sym}_hawkes"].values
        q05 = feat[f"{sym}_hawkes_q05"].values
        q95 = feat[f"{sym}_hawkes_q95"].values
        close = feat[f"{sym}_close"].values

        regime = np.zeros(n, dtype=np.int8)
        anchor_price = np.nan
        current_regime = 0

        for i in range(1, n):
            if not np.isnan(q05[i]) and hawkes[i] < q05[i]:
                anchor_price = close[i]
                current_regime = 0

            # breakout: cross above q95 from below
            if (not np.isnan(q95[i]) and not np.isnan(q95[i-1])
                    and hawkes[i] > q95[i] and hawkes[i-1] <= q95[i-1]
                    and not np.isnan(anchor_price)):
                if close[i] > anchor_price:
                    current_regime = 1
                elif close[i] < anchor_price:
                    current_regime = -1

            regime[i] = current_regime

        feat[f"{sym}_regime"] = regime

    return feat


# sizing logic

def compute_position_size(atr: float, cfg: Config) -> int:
    if not cfg.use_dynamic_sizing or np.isnan(atr) or atr <= 0:
        return 1
    risk_per_contract = atr * cfg.stop_loss_atr_mult * cfg.point_value
    if risk_per_contract <= 0:
        return 1
    return int(max(1, min(round(cfg.target_risk_dollars / risk_per_contract), cfg.max_contracts)))


# execution loop

@dataclass
class Trade:
    entry_time: pd.Timestamp
    direction: int
    entry_price: float
    contracts: int = 1
    exit_time: Optional[pd.Timestamp] = None
    exit_price: float = 0.0
    gross_pnl: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0
    bars_held: int = 0
    exit_reason: str = ""


def _close_trade(trade: Trade, exit_ts, exit_price: float, bars_held: int,
                 reason: str, cfg: Config):
    trade.exit_time = pd.Timestamp(exit_ts)
    trade.exit_price = exit_price
    trade.bars_held = bars_held
    trade.exit_reason = reason
    c = trade.contracts
    pv = cfg.point_value
    slip = cfg.slippage_ticks * cfg.tick_size * pv
    trade.gross_pnl = (trade.exit_price - trade.entry_price) * trade.direction * pv * c
    trade.fees = 2 * cfg.fee_per_side * c + 2 * slip * c
    trade.net_pnl = trade.gross_pnl - trade.fees


def run_backtest(feat: pd.DataFrame, cfg: Config) -> list[Trade]:
    trades: list[Trade] = []
    n = len(feat)
    ts = feat.index.values
    sym = cfg.trade_asset

    nq_open = feat[f"{sym}_open"].values
    nq_close = feat[f"{sym}_close"].values
    nq_atr = feat[f"{sym}_atr"].values
    nq_regime = feat[f"{sym}_regime"].values
    nq_advantage = feat["nq_advantage"].values
    is_rth = feat["is_rth"].values
    use_hawkes = (cfg.mode == "RS_HAWKES")

    in_position = False
    trade: Optional[Trade] = None
    bars_held = 0
    entry_bar = 0
    hawkes_reset_count = 0
    warmup = max(cfg.cmma_long, cfg.hawkes_quantile_window) + 10

    for i in range(warmup, n - 1):
        if in_position:
            bars_held += 1
            exec_price = nq_open[i + 1]

            if nq_regime[i] == 0:
                hawkes_reset_count += 1
            else:
                hawkes_reset_count = 0
            # exit logic while in min hold
            if bars_held < cfg.min_hold_bars:
                atr_entry = nq_atr[entry_bar]
                if not np.isnan(atr_entry) and atr_entry > 0:
                    unrealized = (nq_close[i] - trade.entry_price) * trade.direction
                    if unrealized < -cfg.stop_loss_atr_mult * atr_entry:
                        _close_trade(trade, ts[i+1], exec_price, bars_held, "stop_loss", cfg)
                        trades.append(trade)
                        in_position, trade = False, None
                continue

            exit_signal, reason = False, ""

            # spread decay
            if trade.direction == 1 and nq_advantage[i] < cfg.exit_spread_threshold:
                exit_signal, reason = True, "spread_decay"
            elif trade.direction == -1 and nq_advantage[i] > -cfg.exit_spread_threshold:
                exit_signal, reason = True, "spread_decay"

            # hawkes sustained cooldown
            if use_hawkes and hawkes_reset_count >= cfg.hawkes_exit_cooldown:
                exit_signal, reason = True, "hawkes_cooldown"

            # max hold
            if bars_held >= cfg.max_hold_bars:
                exit_signal, reason = True, "max_hold"

            # stop/target
            atr_entry = nq_atr[entry_bar]
            if not np.isnan(atr_entry) and atr_entry > 0:
                unrealized = (nq_close[i] - trade.entry_price) * trade.direction
                if unrealized < -cfg.stop_loss_atr_mult * atr_entry:
                    exit_signal, reason = True, "stop_loss"
                if unrealized > cfg.take_profit_atr_mult * atr_entry:
                    exit_signal, reason = True, "take_profit"

            if exit_signal:
                _close_trade(trade, ts[i+1], exec_price, bars_held, reason, cfg)
                trades.append(trade)
                in_position, trade = False, None
                continue  # no same-bar re-entry

        else:
            # Entry evaluation
            if cfg.rth_only and not is_rth[i]:
                continue
            if abs(nq_advantage[i]) < cfg.entry_spread_threshold:
                continue

            direction = 1 if nq_advantage[i] > cfg.entry_spread_threshold else -1
            if use_hawkes and nq_regime[i] != direction:
                continue

            trade = Trade(
                entry_time=pd.Timestamp(ts[i + 1]),
                direction=direction,
                entry_price=nq_open[i + 1],
                contracts=compute_position_size(nq_atr[i], cfg),
            )
            in_position = True
            bars_held = 0
            entry_bar = i
            hawkes_reset_count = 0

    # Close any open position at end of data
    if in_position and trade is not None:
        _close_trade(trade, ts[-1], nq_close[-1], bars_held, "end_of_data", cfg)
        trades.append(trade)

    return trades


# metrics

def build_trade_df(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "entry_time": t.entry_time, "exit_time": t.exit_time,
        "direction": t.direction, "entry_price": t.entry_price,
        "exit_price": t.exit_price, "contracts": t.contracts,
        "gross_pnl": t.gross_pnl, "fees": t.fees, "net_pnl": t.net_pnl,
        "bars_held": t.bars_held, "exit_reason": t.exit_reason,
    } for t in trades])


def compute_metrics(trades: list[Trade], feat: pd.DataFrame) -> dict:
    if not trades:
        return {"total_trades": 0}

    pnls = np.array([t.net_pnl for t in trades])
    gross = np.array([t.gross_pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)

    first, last = trades[0].entry_time, trades[-1].exit_time
    n_days = max((last - first).days, 1) if last else 1

    gw, gl = gross[gross > 0].sum(), abs(gross[gross < 0].sum())

    tdf = build_trade_df(trades)
    tdf["date"] = pd.to_datetime(tdf["entry_time"]).dt.date
    daily = tdf.groupby("date")["net_pnl"].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0

    return {
        "total_trades": len(trades),
        "win_rate": len(wins) / len(pnls),
        "avg_trade": pnls.mean(),
        "avg_winner": wins.mean() if len(wins) else 0,
        "avg_loser": losses.mean() if len(losses) else 0,
        "profit_factor": gw / gl if gl > 0 else np.inf,
        "total_net_pnl": pnls.sum(),
        "max_drawdown": (equity - peak).min(),
        "avg_bars_held": np.mean([t.bars_held for t in trades]),
        "trades_per_day": len(trades) / n_days,
        "exposure_pct": sum(t.bars_held * t.contracts for t in trades) / len(feat) * 100,
        "sharpe_annual": sharpe,
        "avg_contracts": np.mean([t.contracts for t in trades]),
    }


def print_metrics(m: dict):
    if m["total_trades"] == 0:
        print("  No trades generated.")
        return
    rows = [
        ("total trades", f"{m['total_trades']}"),
        ("win rate", f"{m['win_rate']:.1%}"),
        ("profit factor", f"{m['profit_factor']:.2f}"),
        ("avg. trade", f"${m['avg_trade']:,.2f}"),
        ("avg. winner", f"${m['avg_winner']:,.2f}"),
        ("avg. losing trade", f"${m['avg_loser']:,.2f}"),
        ("net pnl", f"${m['total_net_pnl']:,.2f}"),
        ("max drawdown", f"${m['max_drawdown']:,.2f}"),
        ("sharpe (annualized)", f"{m['sharpe_annual']:.2f}"),
        ("avg. bars held", f"{m['avg_bars_held']:.1f}"),
        ("trades/day", f"{m['trades_per_day']:.2f}"),
        ("exposure", f"{m['exposure_pct']:.1f}%"),
        ("avg. contracts per trade", f"{m['avg_contracts']:.1f}"),
    ]
    for label, val in rows:
        print(f"  {label:20s} {val}")


# plotting

def plot_results(trades: list[Trade], feat: pd.DataFrame, cfg: Config):
    out = Path(cfg.results_dir)
    out.mkdir(exist_ok=True)
    if not trades:
        return

    tdf = build_trade_df(trades)
    pnls = tdf["net_pnl"].values
    equity = np.cumsum(pnls)
    sym = cfg.trade_asset

    # strategy time series
    tdf_sorted = tdf.sort_values("entry_time")
    strat_times = pd.to_datetime(tdf_sorted["entry_time"].values)
    strat_eq = np.cumsum(tdf_sorted["net_pnl"].values)

    # buy and hold series
    nq_close = feat[f"{sym}_close"]
    first_time = strat_times[0]
    nq_period = nq_close.loc[first_time:]
    bh_pnl = (nq_period - nq_period.iloc[0]) * cfg.point_value

    # overall strategy drawdown
    s_peak = np.maximum.accumulate(strat_eq)
    s_dd = strat_eq - s_peak

    # buy and hold drawdown
    bh_arr = bh_pnl.values
    bh_peak = np.maximum.accumulate(bh_arr)
    bh_dd = bh_arr - bh_peak

    # equity curve plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(strat_times, strat_eq, linewidth=1, color="steelblue")
    ax.set_title("Equity Curve (Net PnL)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "1_equity_curve.png", dpi=150)
    plt.close(fig)

    # compare with buy and hold
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(nq_period.index, bh_arr, linewidth=0.7, color="gray", alpha=0.7,
            label="NQ Buy & Hold (1 ct)")
    ax.plot(strat_times, strat_eq, linewidth=1.2, color="steelblue",
            label="Strategy")
    ax.set_title("Strategy vs NQ Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "2_vs_buyhold.png", dpi=150)
    plt.close(fig)

    # correlation heatmap stuff
    feature_cols = [c for c in feat.columns
                    if "_open" not in c and c != "is_rth"
                    and "_q05" not in c and "_q95" not in c
                    and "_regime" not in c]
    corr = feat[feature_cols].dropna().corr()
    n_feat = len(corr)
    fig, ax = plt.subplots(figsize=(max(14, n_feat * 0.9), max(12, n_feat * 0.75)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_feat))
    ax.set_yticks(range(n_feat))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    for i in range(n_feat):
        for j in range(n_feat):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(out / "3_feature_correlation.png", dpi=150)
    plt.close(fig)

    # drawdown
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(strat_times, s_dd, 0, alpha=0.4, color="red")
    ax.set_title("Strategy Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "4_drawdown.png", dpi=150)
    plt.close(fig)

    # drawdown comparison with buy and hold
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    axes[0].fill_between(nq_period.index, bh_dd, 0, alpha=0.4, color="gray")
    axes[0].set_title("NQ Buy & Hold Drawdown")
    axes[0].set_ylabel("Drawdown ($)")
    axes[0].text(0.02, 0.15, f"Max DD: ${bh_dd.min():,.0f}",
                 transform=axes[0].transAxes, fontsize=11, color="red")
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(strat_times, s_dd, 0, alpha=0.4, color="steelblue")
    axes[1].set_title("Strategy Drawdown")
    axes[1].set_ylabel("Drawdown ($)")
    axes[1].set_xlabel("Date")
    axes[1].text(0.02, 0.15, f"Max DD: ${s_dd.min():,.0f}",
                 transform=axes[1].transAxes, fontsize=11, color="red")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "5_drawdown_comparison.png", dpi=150)
    plt.close(fig)

# main

def main():
    cfg = CFG
    syms = list(cfg.files.keys())
    t0 = _time.time()

    data = load_data(cfg)
    data = align_data(data, cfg.resample_freq)

    feat = engineer_features(data, cfg)
    feat = compute_hawkes_regimes(feat, syms)

    warmup_end = max(cfg.cmma_long, cfg.hawkes_quantile_window) + 10
    feat = feat.iloc[warmup_end:].copy()
    feat = feat.dropna(subset=[f"{s}_hawkes_q95" for s in syms])

    trades = run_backtest(feat, cfg)
    print(f"  {len(trades)} trades in {_time.time() - t0:.1f}s")

    metrics = compute_metrics(trades, feat)
    print_metrics(metrics)
    plot_results(trades, feat, cfg)

    if trades:
        tdf = build_trade_df(trades)
        tdf.to_csv(Path(cfg.results_dir) / "trades.csv", index=False)

if __name__ == "__main__":
    main()
