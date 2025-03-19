import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import shapiro
import random

# ------------------------------
# 1. Load and Prepare the Data (game_data.csv)
# ------------------------------
path = os.path.join(os.path.dirname(__file__), r"datasets/game_data.csv")
df = pd.read_csv(path)

df['Month'] = pd.to_datetime(df['Month'], infer_datetime_format=True)
df = df.sort_values('Month')

df.set_index('Month', inplace=True)
# Ignore the first year of data
df = df[df.index >= df.index.min() + pd.DateOffset(months=6)]
df = df.dropna()
df = df[df['Avg. Players'] > 0]
print(df.describe())
ts = df['Avg. Players'].pct_change() * 100
print(ts.describe())
# Find where percentage change is less than 100% or greater than -100% or not nan

ts = ts[(ts < 100) & (ts > -100)]
ts = ts.dropna()

ts.name = 'value'
ts = ts.dropna()

# ------------------------------
# 2. Apply Detrending Transformations
# ------------------------------
rolling_window = 3  # 24-month rolling window for smoothing
rolling_mean = ts.rolling(rolling_window, center=True).mean()
ts_detrended = ts - rolling_mean
ts_detrended = ts_detrended.dropna()
decomposition = seasonal_decompose(ts_detrended, model='addative', period=12)
ts_detrended = ts_detrended - decomposition.trend - decomposition.seasonal
ts_detrended = ts_detrended.dropna()
print(ts_detrended.describe())

stat, p_value = shapiro(ts_detrended)
print(f"Shapiro-Wilk Test: p-value = {p_value:.8f}")

# ------------------------------
# 3. Load All Events Data ("all_results.csv")
# ------------------------------
all_results_path = os.path.join(os.path.dirname(__file__), r"all_results.csv")
events_df = pd.read_csv(all_results_path, header=0)
events_df['date'] = pd.to_datetime(events_df['date'], unit='s')
events_df['appid'] = events_df['appid'].astype(int)
# We use 'topic' as the event type

# ------------------------------
# 4. Define Helper Functions
# ------------------------------
def process_game_timeseries(game_df, game_id):
    df_game = game_df[game_df['Game_Id'] == game_id].copy()
    df_game = df_game.sort_values('Month')
    initial_count = len(df_game)
    df_game.set_index('Month', inplace=True)
    
    # Filter out the first year of data
    df_game = df_game[df_game.index >= df_game.index.min() + pd.DateOffset(years=1)]
    after_year_filter_count = len(df_game)
    
    ts_local = df_game['Avg. Players'].pct_change() * 100
    ts_local = ts_local.dropna()
    after_pct_change_count = len(ts_local)
    
    rolling_window = 12  # Adjust if needed
    rolling_mean = ts_local.rolling(rolling_window, center=True).mean()
    ts_det = ts_local - rolling_mean
    ts_det = ts_det.dropna()
    after_rolling_count = len(ts_det)
    
    # print(f"Game ID {game_id} - initial: {initial_count}, after year filter: {after_year_filter_count}, "
    #       f"after pct_change: {after_pct_change_count}, after rolling detrending: {after_rolling_count}")
    
    # Check if enough data remains for seasonal decomposition
    if after_rolling_count < 24:
        ts_det.name = 'value'
        return ts_det
    try:
        decomposition = seasonal_decompose(ts_det, model='additive', period=12)
        ts_det = ts_det - decomposition.trend - decomposition.seasonal
        ts_det = ts_det.dropna()
    except ValueError as e:
        pass
    ts_det.name = 'value'
    return ts_det


def post_event_ttest(ts, event_date, window_months=3):
    event_date = pd.to_datetime(event_date)
    post_start = event_date
    post_end = event_date + pd.DateOffset(months=window_months)
    post_data = ts[post_start:post_end]
    baseline_data = ts[(ts.index < post_start) | (ts.index >= post_end)]
    t_stat, p_val = stats.ttest_ind(post_data, baseline_data, equal_var=False)
    return t_stat, p_val, post_data, baseline_data

# ------------------------------
# 5. Loop Over Each Game and Run Per-Game Analysis
# ------------------------------
# This dictionary will store per-game results per topic
# and we'll later aggregate over all games.
all_game_topic_results = {}  # key: topic, value: list of diff values from each game

unique_game_ids = events_df['appid'].unique()
for game_id in unique_game_ids:

    # print(f"\n=== Analysis for Game ID {game_id} ===")
    ts_detrended_game = process_game_timeseries(df.reset_index(), game_id)
    if ts_detrended_game.empty:
        print("No valid time series data for this game. Skipping.")
        continue
    # # (Optional) Plot the time series for this game
    # plt.figure(figsize=(12, 4))
    # plt.plot(ts_detrended_game.index, ts_detrended_game, label='Detrended Series')
    # plt.xlabel("Date")
    # plt.ylabel("Detrended % Change")
    # plt.title(f"Game ID {game_id}: Detrended Time Series")
    # plt.legend()
    # plt.show()
    
    # Filter events for this game
    game_events = events_df[events_df['appid'] == game_id].copy()
    start_date = ts_detrended_game.index.min()
    game_events = game_events[game_events['date'] >= start_date]
    game_events = game_events.sort_values('date')
    if game_events.empty:
        print("No events for this game after start date. Skipping event analysis.")
        continue
    
    window_months = 3  # post-event window in months
    aggregated_post = {}  # key = topic, value = set of indices in ts_detrended_game
    for topic in game_events['topic'].unique():
        aggregated_post[topic] = set()
    for idx, row in game_events.iterrows():
        event_date = row['date']
        topic = row['topic']
        post_window = ts_detrended_game.loc[event_date: event_date + pd.DateOffset(months=window_months)].index
        aggregated_post[topic].update(post_window)
    
    # For each topic, compute t-test and effect (diff)
    for topic, post_indices in aggregated_post.items():
        post_indices = sorted(list(post_indices))
        post_values = ts_detrended_game.loc[post_indices] if len(post_indices) > 0 else pd.Series([], dtype=ts_detrended_game.dtype)
        baseline_indices = ts_detrended_game.index.difference(post_indices)
        baseline_values = ts_detrended_game.loc[baseline_indices]
        if len(post_values) < 2 or len(baseline_values) < 2:
            print(f"Skipping topic {topic} for Game ID {game_id}: insufficient data (post: {len(post_values)}, baseline: {len(baseline_values)})")
            continue
        t_stat, p_val = stats.ttest_ind(post_values, baseline_values, equal_var=False)
        diff = post_values.mean() - baseline_values.mean()
        # Save diff for aggregated analysis
        if topic not in all_game_topic_results:
            all_game_topic_results[topic] = []
        all_game_topic_results[topic].append(diff)
        
        n1, n2 = len(post_values), len(baseline_values)
        s1, s2 = post_values.std(ddof=1), baseline_values.std(ddof=1)
        se_diff = np.sqrt(s1**2/n1 + s2**2/n2)
        df_num = (s1**2/n1 + s2**2/n2)**2
        df_den = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
        df_val = df_num / df_den
        t_crit = stats.t.ppf(1-0.025, df_val)
        ci_lower = diff - t_crit * se_diff
        ci_upper = diff + t_crit * se_diff
        # print(f"Game ID {game_id} | Topic {topic}: n_post = {n1}, n_baseline = {n2}, "
        #       f"Mean Difference = {diff:.2f} (95% CI: {ci_lower:.2f}, {ci_upper:.2f}), "
        #       f"t-stat = {t_stat:.2f}, p-value = {p_val:.7f}")
    
    # # (Optional) Plot the detrended series with event markers for this game
    # plt.figure(figsize=(12, 6))
    # plt.plot(ts_detrended_game.index, ts_detrended_game, label='Detrended Series', color='blue', alpha=0.6)
    # for idx, row in game_events.iterrows():
    #     event_date = pd.to_datetime(row['date'])
    #     topic = row['topic']
    #     color = 'red' if topic == 1 else 'green'
    #     plt.axvline(event_date, color=color, linestyle='dashed', alpha=0.8)
    # plt.xlabel("Date")
    # plt.ylabel("Detrended Value (%)")
    # plt.title(f"Game ID {game_id}: Time Series with Event Markers")
    # plt.legend()
    # plt.show()

# ------------------------------
# 6. Aggregate and Test Significance Across All Games for Each Topic
# ------------------------------
print("\n=== Aggregated Results Over All Games by Topic ===")
aggregated_results = {}
for topic, diffs in all_game_topic_results.items():
    diffs = np.array(diffs)
    if len(diffs) < 2:
        print(f"Topic {topic}: insufficient aggregated data (n={len(diffs)}) for a t-test.")
        continue
    t_stat, p_val = stats.ttest_1samp(diffs, popmean=0)
    aggregated_results[topic] = {
        'n_games': len(diffs),
        'mean_diff': np.mean(diffs),
        'std_diff': np.std(diffs, ddof=1),
        't_stat': t_stat,
        'p_val': p_val
    }
    print(f"Topic {topic}: n_games = {len(diffs)}, Mean Difference = {np.mean(diffs):.2f}, "
          f"t-stat = {t_stat:.2f}, p-value = {p_val:.7f}")

# Optionally, plot the aggregated differences (with standard errors) for each topic
if aggregated_results:
    topics = sorted(aggregated_results.keys())
    mean_diffs = [aggregated_results[t]['mean_diff'] for t in topics]
    std_diffs = [aggregated_results[t]['std_diff'] for t in topics]
    n_games = [aggregated_results[t]['n_games'] for t in topics]
    # Calculate standard error for aggregated diff per topic
    se_agg = [std_diffs[i] / np.sqrt(n_games[i]) for i in range(len(n_games))]
    
    plt.figure(figsize=(10, 6))
    bar_colors = ['green' if md > 0 else 'red' for md in mean_diffs]
    plt.bar([str(t) for t in topics], mean_diffs, yerr=se_agg, capsize=5, color=bar_colors)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel("Event Topic")
    plt.ylabel("Aggregated Mean Difference (%)")
    plt.title("Aggregated Post-Event vs. Baseline Mean Differences Across All Games")
    plt.show()


# # Window size for ITS regression (used earlier)
# window_size = 3

# # --------------------------------------------
# # 4. Helper Functions for ITS Analysis
# # --------------------------------------------
# def get_window_data(ts, event_date, window_size):
#     event_date = pd.to_datetime(event_date)
#     start_date = event_date - pd.DateOffset(months=window_size)
#     end_date = event_date + pd.DateOffset(months=window_size)
#     window_data = ts[start_date:end_date].copy().reset_index()
#     window_data.rename(columns={'index': 'Month', ts.name: 'value'}, inplace=True)
#     window_data['time'] = (window_data['Month'] - event_date).dt.days / 30.0
#     return window_data

# def its_regression(window_data):
#     data = window_data.copy()
#     if data.empty:
#         print("window_data is empty, skipping regression.")
#         return None  # Skip regression if no data
#     data['D'] = (data['time'] >= 0).astype(int)
#     data['time_after'] = data['time'] * data['D']
#     X = data[['time', 'D', 'time_after']]
#     y = data['value']
#     if X.empty or y.empty:
#         print("Error: X or y is empty for regression.")
#         return None
#     X = sm.add_constant(X).dropna()
#     y = y.dropna()
#     if X.empty or y.empty:
#         print("Error after dropping NaNs: X or y is still empty.")
#         return None
#     model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
#     return model

# # ------------------------------
# # 5. Loop Through Events and Run ITS Analysis
# # ------------------------------
# results = {}
# cum_effect_list = []

# for event in events:
#     event_date = event['date']
#     event_type = event['type']
#     window_data = get_window_data(ts_detrended, event_date, window_size)
#     if window_data.empty:
#         continue
#     its_model = its_regression(window_data)
#     if its_model is None:
#         continue  # Skip empty results
#     effect = its_model.params['D']
#     p_val = its_model.pvalues['D']
#     results[event_date] = {'ITS': its_model}
#     cum_effect_list.append({
#         'event_date': pd.to_datetime(event_date),
#         'effect': effect,
#         'p_value': p_val,
#         'event_type': event_type
#     })

# # ------------------------------
# # 6. Aggregate ITS Results for Category Testing
# # ------------------------------
# cum_df = pd.DataFrame(cum_effect_list)
# for etype in cum_df['event_type'].unique():
#     subset = cum_df[cum_df['event_type'] == etype]
#     effects = subset['effect'].dropna()
#     n = len(effects)
#     if n > 1:
#         t_stat, p_val = stats.ttest_1samp(effects, popmean=0)
#         print(f"\nCategory {etype}: n = {n}, Mean Effect = {effects.mean():.2f}%, t-stat = {t_stat:.2f}, p-value = {p_val:.7f}")
#     else:
#         print(f"\nCategory {etype}: n = {n} (not enough samples for t-test)")
# cum_df = cum_df.sort_values('event_date')

# plt.figure(figsize=(12, 6))
# plt.plot(cum_df['event_date'], cum_df['p_value'], marker='o', linestyle='-', label='p-value')
# plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Threshold (0.05)')
# plt.axhline(y=0.01, color='g', linestyle='--', label='Highly Significant (0.01)')
# plt.axhline(y=0.1, color='b', linestyle='--', label='Weak Significance (0.1)')
# plt.xlabel("Event Date")
# plt.ylabel("p-value")
# plt.title("Statistical Significance (p-values) of ITS Effects")
# plt.legend()
# plt.xticks(rotation=45)
# plt.grid()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(cum_df['event_date'], cum_df['effect'], marker='o', linestyle='-', label='Effect Size (%)')
# plt.axhline(y=0, color='k', linestyle='--', label='No Effect')
# plt.xlabel("Event Date")
# plt.ylabel("Effect Size (%)")
# plt.title("Interrupted Time Series Effect Sizes Over Time")
# plt.legend()
# plt.xticks(rotation=45)
# plt.grid()
# plt.show()