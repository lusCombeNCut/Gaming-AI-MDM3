import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import shapiro
from scipy.stats import t  # Import for confidence intervals

# Set global font size for all plots
plt.rcParams.update({'font.size': 15})

# ------------------------------
# 1. Load and Prepare Global Data
# ------------------------------
data_path = os.path.join(os.path.dirname(__file__), "datasets/game_data.csv")
df = pd.read_csv(data_path, parse_dates=['Month'], infer_datetime_format=True)
df.sort_values('Month', inplace=True)
df.set_index('Month', inplace=True)

# Remove early data and bad values
df = df[df.index >= df.index.min() + pd.DateOffset(months=6)]
df = df.dropna()
df = df[df['Avg. Players'] > 0]

# Compute overall percentage change and remove extreme values (>|100|%)
global_pct = df['Avg. Players'].pct_change() * 100
global_pct.replace([np.inf, -np.inf], np.nan, inplace=True)
global_pct = global_pct.dropna()
global_pct = global_pct[(global_pct < 100) & (global_pct > -100)]
# Ensure unique dates
global_pct = global_pct[~global_pct.index.duplicated(keep='first')]

# Detrend global series using seasonal decomposition (no rolling mean)
def detrend_series(ts, period):
    ts = ts.dropna()
    try:
        decomposition = seasonal_decompose(ts, model='additive', period=period)
        ts_det = ts - decomposition.trend - decomposition.seasonal
        ts_det = ts_det.dropna()
    except ValueError:
        ts_det = ts
    return ts_det

global_detrended = detrend_series(global_pct, period=12)
print(global_detrended.describe())

stat, p_val = shapiro(global_detrended)
print(f"Shapiro-Wilk Test (global detrended): p-value = {p_val:.8f}")

# ------------------------------
# 2. Load Events Data (all_results.csv)
# ------------------------------
results_path = os.path.join(os.path.dirname(__file__), "all_results.csv")
events_df = pd.read_csv(results_path)
events_df['date'] = pd.to_datetime(events_df['date'], unit='s')
events_df['appid'] = events_df['appid'].astype(int)

# ------------------------------
# 3. Helper Functions for Per-Game Processing
# ------------------------------
def process_game_timeseries(game_df, game_id, pct_upper=100, pct_lower=-100):
    df_game = game_df[game_df['Game_Id'] == game_id].copy()
    df_game.sort_values('Month', inplace=True)
    df_game.set_index('Month', inplace=True)
    df_game = df_game[df_game.index >= df_game.index.min() + pd.DateOffset(years=1)]
    if df_game.empty:
        return pd.Series(dtype=float)

    ts = df_game['Avg. Players'].pct_change() * 100
    ts.replace([np.inf, -np.inf], np.nan, inplace=True)
    ts = ts.dropna()
    ts = ts[(ts < pct_upper) & (ts > pct_lower)]
    if ts.empty:
        return ts

    if len(ts) >= 24:
        try:
            decomposition = seasonal_decompose(ts, model='additive', period=12)
            ts_det = ts - decomposition.trend - decomposition.seasonal
            ts_det = ts_det.dropna()
        except ValueError:
            ts_det = ts
    else:
        ts_det = ts
    return ts_det

def compute_event_effect(ts, event_dates, window_months=3):
    post_indices = set()
    for event_date in event_dates:
        # Only consider the post-update window.
        start = event_date
        end = event_date + pd.DateOffset(months=window_months)
        window_idx = ts.loc[start:end].index
        post_indices.update(window_idx)
    post_indices = sorted(list(post_indices))
    post_data = ts.loc[post_indices] if post_indices else pd.Series(dtype=float)
    baseline_data = ts.loc[ts.index.difference(post_indices)]
    return post_data, baseline_data

# ------------------------------
# 4. Per-Game Analysis & Aggregated Results (across all games)
# ------------------------------
all_game_topic_results = {}
topic_agg_counts = {}

unique_game_ids = events_df['appid'].unique()
for game_id in unique_game_ids:
    ts_det = process_game_timeseries(df.reset_index(), game_id)
    if ts_det.empty:
        continue

    game_events = events_df[(events_df['appid'] == game_id) & (events_df['date'] >= ts_det.index.min())].sort_values('date')
    if game_events.empty:
        continue

    topics = game_events['topic'].unique()
    for topic in topics:
        topic_events = game_events[game_events['topic'] == topic]
        event_dates = pd.to_datetime(topic_events['date'])
        post_data, baseline_data = compute_event_effect(ts_det, event_dates)

        if len(post_data) < 2 or len(baseline_data) < 2:
            continue

        diff = post_data.mean() - baseline_data.mean()
        all_game_topic_results.setdefault(topic, []).append(diff)

        if topic not in topic_agg_counts:
            topic_agg_counts[topic] = {'post': 0, 'baseline': 0}
        topic_agg_counts[topic]['post'] += len(post_data)
        topic_agg_counts[topic]['baseline'] += len(baseline_data)

print("\n=== Aggregated Results Over All Games by Topic ===")
aggregated_results = {}
confidence_level = 0.90

for topic, diffs in all_game_topic_results.items():
    diffs = np.array(diffs)
    if len(diffs) < 2:
        continue
    t_stat, p_val = stats.ttest_1samp(diffs, popmean=0)
    n_games = len(diffs)
    total_post = topic_agg_counts[topic]['post']
    total_baseline = topic_agg_counts[topic]['baseline']
    
    std_diff = np.std(diffs, ddof=1)
    sem = std_diff / np.sqrt(n_games)
    t_critical = t.ppf((1 + confidence_level) / 2, df=n_games - 1)
    ci_90 = t_critical * sem

    aggregated_results[topic] = {
        'n_games': n_games,
        'mean_diff': np.mean(diffs),
        'std_diff': std_diff,
        't_stat': t_stat,
        'p_val': p_val,
        'total_post_samples': total_post,
        'total_baseline_samples': total_baseline,
        'ci_90': ci_90
    }
    
    print(f"Topic {topic}: n_games = {n_games}, Total Post Samples = {total_post}, Total Baseline Samples = {total_baseline}, "
          f"Mean Diff = {np.mean(diffs):.2f}, 90% CI = ±{ci_90:.2f}, t-stat = {t_stat:.2f}, p-value = {p_val:.7f}")

if aggregated_results:
    topics = sorted(aggregated_results.keys())
    mean_diffs = [aggregated_results[t]['mean_diff'] for t in topics]
    ci_90_values = [aggregated_results[t]['ci_90'] for t in topics]

    plt.figure(figsize=(10, 6))
    bar_colors = ['green' if m > 0 else 'red' for m in mean_diffs]
    plt.bar([str(t) for t in topics], mean_diffs, capsize=5, color=bar_colors)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel("Event Topic")
    plt.ylabel("Mean Difference (%)")
    plt.title("Aggregated Post-Event vs Baseline Mean Differences") 
    plt.show()

# =============================================================================
# Additional Plots for PowerPoint Presentation (for a Specific Game)
# =============================================================================

# Select a specific game (the first one with sufficient data and events)
example_game_id = None
for game_id in unique_game_ids:
    ts_det_example = process_game_timeseries(df.reset_index(), game_id)
    if not ts_det_example.empty:
        game_events = events_df[(events_df['appid'] == game_id) &
                                 (events_df['date'] >= ts_det_example.index.min())].sort_values('date')
        if not game_events.empty:
            example_game_id = game_id
            break

if example_game_id is None:
    print("No example game found with sufficient data for additional plots.")
else:
    # Prepare data for the specific game
    df_game = df.reset_index()
    df_game = df_game[df_game['Game_Id'] == example_game_id].copy()
    df_game.sort_values('Month', inplace=True)
    df_game.set_index('Month', inplace=True)
    df_game = df_game[df_game.index >= df_game.index.min() + pd.DateOffset(years=1)]
    
    raw_ts = df_game['Avg. Players'].pct_change() * 100
    raw_ts.replace([np.inf, -np.inf], np.nan, inplace=True)
    raw_ts = raw_ts.dropna()
    raw_ts = raw_ts[(raw_ts < 100) & (raw_ts > -100)]
    raw_ts = raw_ts[~raw_ts.index.duplicated(keep='first')]
    
    # Perform seasonal decomposition directly on the raw series
    try:
        decomposition = seasonal_decompose(raw_ts, model='additive', period=12)
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()
        final_processed = (raw_ts - decomposition.trend - decomposition.seasonal).dropna()
    except Exception as e:
        print("Seasonal decomposition failed for example game:", e)
        trend = raw_ts.copy()
        seasonal = raw_ts.copy()
        final_processed = raw_ts.copy()
    
    # --- Additional Plot 1: Static Plot for Seasonal Decomposition (for Specific Game) ---
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(14, 12))

    axs[0].plot(raw_ts.index, raw_ts)
    axs[0].set_ylabel('Raw\n(% Change)')
    axs[0].grid(True)

    if not trend.empty:
        axs[1].plot(trend.index, trend, color='blue')
        axs[1].set_ylabel('Trend\n(% Change)')
        axs[1].grid(True)

    if not seasonal.empty:
        axs[2].plot(seasonal.index, seasonal, color='red')
        axs[2].set_ylabel('Seasonal\n(% Change)')
        axs[2].grid(True)

    axs[3].plot(final_processed.index, final_processed, color='green')
    axs[3].set_ylabel('Final\n(% Change)')
    axs[3].grid(True)

    plt.xlabel('Month')
    plt.suptitle(f'Average Monthly Player Timeseries Decomposition for Game - {example_game_id} (Mapel Story)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
            