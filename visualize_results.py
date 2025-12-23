"""
Visualization Script for CBM QR-DQN Training Results (v2.0)

Advanced visualization features based on base_markov-dqn-v09:
- Comprehensive training progress plots (reward, loss, cost)
- QR-DQN quantile distribution analysis
- Risk analysis (VaR, CVaR, quantile statistics)
- State transition analysis
- Action distribution analysis
- Policy evaluation on test episodes
- Distribution shape and uncertainty analysis
- Equipment aging analysis visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torch
import sys
from scipy import stats
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cbm_environment import EquipmentCBMEnvironment, STATE_NAMES, ACTION_NAMES
from train_cbm_dqn_v2 import CBMQRDQN
from data_preprocessor import CBMDataPreprocessor


def plot_training_history(history_path: Path, save_dir: Path):
    """Plot comprehensive training history with advanced metrics"""
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    episode_rewards = history['episode_rewards']
    episode_lengths = history['episode_lengths']
    losses = history['losses']
    
    episodes = len(episode_rewards)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    
    # Compute moving averages
    window = 50
    def moving_average(data, w):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w)/w, mode='valid')
    
    ma_rewards = moving_average(episode_rewards, window)
    ma_losses = moving_average(losses, min(window, len(losses)//10)) if len(losses) > 0 else []
    
    # 1. Episode Rewards
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(episode_rewards, alpha=0.3, label='Raw', color='red')
    if len(ma_rewards) >= window:
        ax1.plot(range(window-1, episodes), ma_rewards, 
                linewidth=2, label=f'MA({window})', color='darkred')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('CBM QR-DQN v2.0: Episode Rewards\n(Quantile Regression, 51 quantiles)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Length
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(episode_lengths, alpha=0.3, label='Raw', color='blue')
    ma_lengths = moving_average(episode_lengths, window)
    if len(ma_lengths) >= window:
        ax2.plot(range(window-1, len(episode_lengths)), ma_lengths, 
                linewidth=2, label=f'MA({window})', color='darkblue')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Length\n(Horizon: 100 steps)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Loss (Quantile Huber Loss)
    ax3 = plt.subplot(2, 3, 3)
    if len(losses) > 0:
        ax3.plot(losses, alpha=0.3, color='orange', label='Raw')
        if len(ma_losses) > 0:
            offset = len(losses) - len(ma_losses)
            ax3.plot(range(offset, len(losses)), ma_losses, 
                    color='darkorange', linewidth=2, label='MA')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Quantile Huber Loss')
        ax3.set_title('Training Loss (QR-DQN, κ=1.0)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    else:
        ax3.text(0.5, 0.5, 'No loss data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Training Loss')
    
    # 4. Learning Progress (Rolling Average)
    ax4 = plt.subplot(2, 3, 4)
    recent_window = min(100, len(episode_rewards) // 4)
    if recent_window > 0:
        recent_rewards = [np.mean(episode_rewards[max(0, i-recent_window):i+1]) 
                         for i in range(len(episode_rewards))]
        ax4.plot(recent_rewards, color='green', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel(f'Average Reward (last {recent_window} eps)')
        ax4.set_title('Learning Progress\n(Rolling Average)')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 5. Reward Distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(episode_rewards, bins=50, alpha=0.7, edgecolor='black', color='purple')
    ax5.axvline(np.mean(episode_rewards), color='red', 
               linestyle='--', linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
    ax5.axvline(np.median(episode_rewards), color='orange',
               linestyle='--', linewidth=2, label=f'Median: {np.median(episode_rewards):.2f}')
    ax5.set_xlabel('Total Reward')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Reward Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate statistics
    final_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    max_reward = np.max(episode_rewards)
    min_reward = np.min(episode_rewards)
    final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else (np.mean(losses) if len(losses) > 0 else 0)
    
    summary_text = f"""
    CBM QR-DQN v2.0 Training Summary
    ═══════════════════════════════════
    
    Training Configuration:
    • Episodes: {episodes}
    • Quantiles: 51
    • Kappa (Huber): 1.0
    • Parallel Envs: 16
    
    Environment:
    • States: 2 (Normal, Anomalous)
    • Actions: 3 (DoNothing, Repair, Replace)
    • Horizon: 100 steps
    
    Performance (Last 100 episodes):
    • Avg Reward: {final_reward:.2f}
    • Avg Loss: {final_loss:.4f}
    
    Overall Statistics:
    • Max Reward: {max_reward:.2f}
    • Min Reward: {min_reward:.2f}
    • Mean Reward: {np.mean(episode_rewards):.2f}
    • Std Reward: {np.std(episode_rewards):.2f}
    
    Optimizations:
    ✓ QR-DQN (Quantile Regression)
    ✓ Noisy Networks
    ✓ Dueling DQN
    ✓ PER (Prioritized Replay)
    ✓ N-step Learning (n=3)
    ✓ Mixed Precision Training
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / "training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Training history plot saved: {save_path}")
    plt.close()


def plot_aging_analysis(preprocessor: CBMDataPreprocessor, equipment_id: int, 
                       measurement_id: int, save_dir: Path):
    """Plot equipment aging analysis"""
    
    # Get processed data
    timeseries_df = preprocessor.extract_timeseries(equipment_id, measurement_id, include_age=True)
    
    if '設備経過年数' not in timeseries_df.columns or timeseries_df['設備経過年数'].isna().all():
        print("⚠️ No aging data available for visualization")
        return
    
    # Apply state labeling to get condition column
    timeseries_df = preprocessor.label_states(timeseries_df)
    
    # Remove invalid age data
    valid_age_df = timeseries_df.dropna(subset=['設備経過年数'])
    
    if len(valid_age_df) == 0:
        print("⚠️ No valid aging data for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Equipment Aging Analysis', fontsize=16, fontweight='bold')
    
    # 1. Age distribution
    axes[0, 0].hist(valid_age_df['設備経過年数'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Equipment Age Distribution')
    axes[0, 0].set_xlabel('Age (years)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Age vs Anomaly Rate
    age_bins = np.linspace(0, valid_age_df['設備経過年数'].max() + 1, 11)
    valid_age_df['age_bin'] = pd.cut(valid_age_df['設備経過年数'], bins=age_bins, include_lowest=True)
    age_anomaly = valid_age_df.groupby('age_bin')['condition'].agg(['mean', 'count']).reset_index()
    age_anomaly = age_anomaly[age_anomaly['count'] >= 10]  # Filter small groups
    
    if len(age_anomaly) > 0:
        bin_centers = [interval.mid for interval in age_anomaly['age_bin']]
        axes[0, 1].plot(bin_centers, age_anomaly['mean'] * 100, 'o-', color='red', linewidth=2, markersize=6)
        axes[0, 1].set_title('Anomaly Rate vs Equipment Age')
        axes[0, 1].set_xlabel('Age (years)')
        axes[0, 1].set_ylabel('Anomaly Rate (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Fit trend line
        if len(bin_centers) > 1:
            z = np.polyfit(bin_centers, age_anomaly['mean'] * 100, 1)
            p = np.poly1d(z)
            axes[0, 1].plot(bin_centers, p(bin_centers), "--", alpha=0.8, color='darkred')
            slope_per_year = z[0]
            axes[0, 1].text(0.05, 0.95, f'Trend: +{slope_per_year:.2f}%/year',
                           transform=axes[0, 1].transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 3. Measurement values over time colored by age
    scatter = axes[1, 0].scatter(valid_age_df.index, valid_age_df['実測値'], 
                                c=valid_age_df['設備経過年数'], cmap='viridis', alpha=0.6, s=10)
    axes[1, 0].set_title('Measurement Values over Time (colored by age)')
    axes[1, 0].set_xlabel('Time Index')
    axes[1, 0].set_ylabel('Measurement Value')
    plt.colorbar(scatter, ax=axes[1, 0], label='Age (years)')
    
    # Add threshold lines
    if '上限値Smax' in valid_age_df.columns and not pd.isna(valid_age_df['上限値Smax'].iloc[0]):
        axes[1, 0].axhline(y=valid_age_df['上限値Smax'].iloc[0], color='red', linestyle='--', alpha=0.7, label='Smax')
    if '下限値Smin' in valid_age_df.columns and not pd.isna(valid_age_df['下限値Smin'].iloc[0]):
        axes[1, 0].axhline(y=valid_age_df['下限値Smin'].iloc[0], color='red', linestyle='--', alpha=0.7, label='Smin')
    axes[1, 0].legend()
    
    # 4. Age correlation analysis
    correlation = valid_age_df[['設備経過年数', 'condition', '実測値']].corr()
    im = axes[1, 1].imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 1].set_title('Correlation Matrix')
    axes[1, 1].set_xticks([0, 1, 2])
    axes[1, 1].set_xticklabels(['Age', 'Condition', 'Measurement'], rotation=45)
    axes[1, 1].set_yticks([0, 1, 2])
    axes[1, 1].set_yticklabels(['Age', 'Condition', 'Measurement'])
    
    # Add correlation values as text
    for i in range(3):
        for j in range(3):
            text = axes[1, 1].text(j, i, f'{correlation.iloc[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=axes[1, 1], label='Correlation')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'aging_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Aging analysis plot saved")


def analyze_qr_distribution(
    env: EquipmentCBMEnvironment,
    policy_net: CBMQRDQN,
    save_dir: Path,
    n_samples: int = 1000,
    device: str = 'cpu'
):
    """
    Analyze QR-DQN return distribution statistics.
    Based on base_markov-dqn-v09 analysis tools.
    """
    
    print("\n" + "="*80)
    print("QR-DQN RETURN DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Sample multiple states for analysis
    states = []
    for _ in range(n_samples):
        state, _ = env.reset()
        states.append(state)
    states = np.array(states, dtype=np.float32)
    
    # Get quantile values for all samples
    with torch.no_grad():
        states_t = torch.FloatTensor(states).to(device)
        q_values, quantiles = policy_net(states_t)
    
    q_values = q_values.cpu().numpy()  # [n_samples, n_actions]
    quantiles = quantiles.cpu().numpy()  # [n_samples, n_actions, n_quantiles]
    
    n_quantiles = quantiles.shape[-1]
    
    # ===== 1. Per-Action Distribution Statistics =====
    print("\n1. PER-ACTION DISTRIBUTION STATISTICS")
    print("-" * 80)
    
    stats_data = []
    
    for a_idx, action_name in enumerate(ACTION_NAMES):
        # Get all quantile values for this action
        action_quantiles = quantiles[:, a_idx, :]  # [n_samples, n_quantiles]
        
        # Compute statistics from quantiles
        means = action_quantiles.mean(axis=1)  # Mean of quantiles per sample
        
        # Variance from quantiles
        variances = ((action_quantiles - means[:, np.newaxis])**2).mean(axis=1)
        stds = np.sqrt(variances)
        
        # Quantile-based risk measures
        q_05 = action_quantiles[:, max(0, int(0.05 * n_quantiles))]  # 5th percentile
        q_25 = action_quantiles[:, max(0, int(0.25 * n_quantiles))]  # 25th percentile
        q_50 = action_quantiles[:, max(0, int(0.50 * n_quantiles))]  # Median
        q_75 = action_quantiles[:, min(n_quantiles-1, int(0.75 * n_quantiles))]  # 75th percentile
        q_95 = action_quantiles[:, min(n_quantiles-1, int(0.95 * n_quantiles))]  # 95th percentile
        
        # VaR and CVaR
        var_5 = np.percentile(means, 5)  # Value at Risk (5%)
        cvar_5 = np.mean(means[means <= var_5])  # Conditional VaR
        
        print(f"\n{action_name}:")
        print(f"  Mean Return:    {np.mean(means):8.2f} ± {np.std(means):6.2f}")
        print(f"  Std Dev:        {np.mean(stds):8.2f} ± {np.std(stds):6.2f}")
        print(f"  Q05 (5%ile):    {np.mean(q_05):8.2f}")
        print(f"  Q25 (25%ile):   {np.mean(q_25):8.2f}")
        print(f"  Q50 (median):   {np.mean(q_50):8.2f}")
        print(f"  Q75 (75%ile):   {np.mean(q_75):8.2f}")
        print(f"  Q95 (95%ile):   {np.mean(q_95):8.2f}")
        print(f"  VaR (5%):       {var_5:8.2f}")
        print(f"  CVaR (5%):      {cvar_5:8.2f}")
        
        stats_data.append({
            'action': action_name,
            'means': means,
            'stds': stds,
            'q_05': q_05,
            'q_25': q_25,
            'q_50': q_50,
            'q_75': q_75,
            'q_95': q_95,
            'var_5': var_5,
            'cvar_5': cvar_5,
            'quantiles': action_quantiles
        })
    
    # ===== 2. Distribution Shape Analysis =====
    print("\n\n2. DISTRIBUTION SHAPE ANALYSIS")
    print("-" * 80)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for a_idx, (action_name, ax) in enumerate(zip(ACTION_NAMES, axes)):
        data = stats_data[a_idx]
        
        # Plot mean distribution
        ax.hist(data['means'], bins=50, alpha=0.7, color=f'C{a_idx}', edgecolor='black')
        ax.axvline(np.mean(data['means']), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(data["means"]):.2f}')
        ax.axvline(data['var_5'], color='orange', linestyle='--',
                   linewidth=2, label=f'VaR(5%): {data["var_5"]:.2f}')
        
        ax.set_xlabel('Expected Return')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{action_name}\n(σ={np.mean(data["stds"]):.2f}, median={np.mean(data["q_50"]):.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_dir / "distribution_statistics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Distribution statistics plot saved: {plot_path}")
    plt.close()
    
    # ===== 3. Uncertainty Analysis =====
    print("\n\n3. UNCERTAINTY ANALYSIS")
    print("-" * 80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Variance comparison
    variances = [np.mean(data['stds'])**2 for data in stats_data]
    ax1.bar(ACTION_NAMES, variances, color=['C0', 'C1', 'C2'])
    ax1.set_ylabel('Variance')
    ax1.set_title('Return Variance by Action\n(Higher = More Uncertain)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # IQR (Interquartile Range) comparison
    iqrs = [np.mean(data['q_75'] - data['q_25']) for data in stats_data]
    ax2.bar(ACTION_NAMES, iqrs, color=['C0', 'C1', 'C2'])
    ax2.set_ylabel('IQR (Q75 - Q25)')
    ax2.set_title('Interquartile Range by Action\n(Higher = More Spread Out)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = save_dir / "uncertainty_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Uncertainty analysis plot saved: {plot_path}")
    plt.close()
    
    # ===== 4. Risk Profile =====
    print("\n\n4. RISK PROFILE")
    print("-" * 80)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(ACTION_NAMES))
    width = 0.25
    
    means = [np.mean(data['means']) for data in stats_data]
    var_5s = [data['var_5'] for data in stats_data]
    cvar_5s = [data['cvar_5'] for data in stats_data]
    
    ax.bar(x - width, means, width, label='Mean Return', color='green', alpha=0.7)
    ax.bar(x, var_5s, width, label='VaR (5%)', color='orange', alpha=0.7)
    ax.bar(x + width, cvar_5s, width, label='CVaR (5%)', color='red', alpha=0.7)
    
    ax.set_xlabel('Action')
    ax.set_ylabel('Return Value')
    ax.set_title('Risk Profile by Action\n(VaR = Value at Risk, CVaR = Conditional Value at Risk)')
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_NAMES)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = save_dir / "risk_profile.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Risk profile plot saved: {plot_path}")
    plt.close()
    
    # ===== 5. Quantile Distribution Visualization =====
    print("\n\n5. QUANTILE DISTRIBUTION VISUALIZATION")
    print("-" * 80)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Sample a few states to show their quantile distributions
    sample_indices = np.random.choice(n_samples, size=min(10, n_samples), replace=False)
    tau = np.linspace(0, 1, n_quantiles)  # Quantile fractions
    
    for a_idx, (action_name, ax) in enumerate(zip(ACTION_NAMES, axes)):
        data = stats_data[a_idx]
        
        # Plot quantile curves for sample states
        for idx in sample_indices:
            quantile_vals = data['quantiles'][idx]
            ax.plot(tau, quantile_vals, alpha=0.3, color=f'C{a_idx}')
        
        # Plot mean quantile curve
        mean_quantiles = data['quantiles'].mean(axis=0)
        ax.plot(tau, mean_quantiles, color='black', linewidth=3, 
               label=f'Mean (Q={np.mean(data["means"]):.2f})')
        
        ax.set_xlabel('Quantile (τ)')
        ax.set_ylabel('Return Value')
        ax.set_title(f'{action_name}\nQuantile Functions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = save_dir / "quantile_distributions.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Quantile distribution plot saved: {plot_path}")
    plt.close()
    
    print("\n" + "="*80)
    print("DISTRIBUTION ANALYSIS COMPLETE")
    print("="*80)
    
    return stats_data


def evaluate_policy(
    env: EquipmentCBMEnvironment,
    policy_net: CBMQRDQN,
    n_episodes: int = 10,
    device: str = 'cpu'
):
    """Evaluate trained policy"""
    
    episode_data = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_temps = []
        
        for step in range(env.horizon):
            # Select action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values, _ = policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()
            
            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_states.append(info['old_condition'])
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_temps.append(info['temperature'])
            
            state = next_state
            
            if terminated or truncated:
                break
        
        episode_data.append({
            'states': episode_states,
            'actions': episode_actions,
            'rewards': episode_rewards,
            'temperatures': episode_temps,
            'total_reward': sum(episode_rewards)
        })
    
    return episode_data


def plot_policy_evaluation(episode_data: list, save_dir: Path):
    """Plot policy evaluation results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Action distribution
    all_actions = []
    for ep in episode_data:
        all_actions.extend(ep['actions'])
    
    action_counts = [all_actions.count(i) for i in range(3)]
    axes[0, 0].bar(ACTION_NAMES, action_counts, alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Action Distribution')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # State distribution
    all_states = []
    for ep in episode_data:
        all_states.extend(ep['states'])
    
    state_counts = [all_states.count(i) for i in range(2)]
    axes[0, 1].bar(STATE_NAMES, state_counts, alpha=0.7, edgecolor='black', color=['green', 'red'])
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('State Distribution')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Sample episode trajectory
    sample_ep = episode_data[0]
    steps = range(len(sample_ep['states']))
    
    ax1 = axes[1, 0]
    ax2 = ax1.twinx()
    
    # State trajectory
    state_line = ax1.step(steps, sample_ep['states'], where='post', 
                          label='State', linewidth=2, color='blue')
    ax1.set_ylabel('State (0=Normal, 1=Anomalous)', color='blue')
    ax1.set_xlabel('Step')
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(STATE_NAMES)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Action markers
    action_steps = []
    repair_steps = []
    replace_steps = []
    for i, action in enumerate(sample_ep['actions']):
        if action == 1:  # Repair
            repair_steps.append(i)
        elif action == 2:  # Replace
            replace_steps.append(i)
    
    if repair_steps:
        ax1.scatter(repair_steps, [0.5]*len(repair_steps), 
                   marker='^', s=100, color='orange', label='Repair', zorder=5)
    if replace_steps:
        ax1.scatter(replace_steps, [0.5]*len(replace_steps), 
                   marker='s', s=100, color='red', label='Replace', zorder=5)
    
    # Temperature trajectory
    temp_line = ax2.plot(steps, sample_ep['temperatures'], 
                         label='Temperature', linewidth=2, color='red', alpha=0.7)
    ax2.set_ylabel('Temperature (°C)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.set_title('Sample Episode Trajectory')
    ax1.grid(True, alpha=0.3)
    
    # Total rewards per episode
    total_rewards = [ep['total_reward'] for ep in episode_data]
    axes[1, 1].bar(range(len(total_rewards)), total_rewards, alpha=0.7, edgecolor='black')
    axes[1, 1].axhline(np.mean(total_rewards), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {np.mean(total_rewards):.2f}')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Total Reward')
    axes[1, 1].set_title('Episode Rewards')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / "policy_evaluation.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Policy evaluation plot saved: {save_path}")
    plt.close()


def plot_state_transition_matrix(transition_matrix: np.ndarray, save_dir: Path):
    """Plot state transition matrix heatmap"""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(transition_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(STATE_NAMES)
    ax.set_yticklabels(STATE_NAMES)
    
    # Labels
    ax.set_xlabel('Next State', fontsize=12)
    ax.set_ylabel('Current State', fontsize=12)
    ax.set_title('State Transition Matrix (DoNothing)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{transition_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transition Probability', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / "transition_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Transition matrix plot saved: {save_path}")
    plt.close()


def main():
    """Main visualization function with comprehensive QR-DQN analysis"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize CBM QR-DQN Results v2.0')
    parser.add_argument('--output_dir', type=str, default='outputs_cbm_v2', help='Output directory')
    parser.add_argument('--equipment_id', type=int, default=43175, help='Equipment ID')
    parser.add_argument('--measurement_id', type=int, default=167473, help='Measurement ID')
    parser.add_argument('--analyze_dist', action='store_true', 
                       help='Perform detailed QR-DQN distribution analysis')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples for distribution analysis')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("="*80)
    print("📊 CBM QR-DQN VISUALIZATION v2.0")
    print("="*80)
    
    # Plot training history
    history_path = output_dir / "training_history.json"
    if history_path.exists():
        print("\n📈 Plotting training history...")
        plot_training_history(history_path, output_dir)
    else:
        print(f"\n⚠️ Training history not found: {history_path}")
    
    # Load trained model
    model_path = output_dir / "policy_net.pth"
    if model_path.exists():
        print("\n🤖 Loading trained model...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy_net = CBMQRDQN(state_dim=3, n_actions=3, n_quantiles=51).to(device)
        policy_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        policy_net.eval()
        
        print("✅ Model loaded")
        
        # Load preprocessed data
        from data_preprocessor import CBMDataPreprocessor
        
        data_dir = Path(__file__).parent / "data" / "private_benchmark"
        preprocessor = CBMDataPreprocessor(data_dir)
        preprocessor.load_data()
        
        stats = preprocessor.process_equipment(args.equipment_id, args.measurement_id)
        
        # Create environment
        transition_matrix = np.array(stats['transition_matrix'], dtype=np.float32)
        temp_min = stats['value_stats']['min']
        temp_max = stats['value_stats']['max']
        normal_temp_min = stats['thresholds']['Smin']
        normal_temp_max = stats['thresholds']['Smax']
        
        # Extract aging parameters from stats if available
        equipment_age = 0.0
        aging_factor = 0.015
        if 'age_analysis' in stats and stats['age_analysis']['age_available']:
            age_range = stats['age_analysis']['age_range']
            equipment_age = np.mean(age_range) if age_range else 0.0
        
        env = EquipmentCBMEnvironment(
            transition_matrix=transition_matrix,
            temperature_range=(temp_min, temp_max),
            normal_temp_range=(normal_temp_min, normal_temp_max),
            horizon=100,
            gamma=0.95,
            equipment_age=equipment_age,
            aging_factor=aging_factor,
            seed=42
        )
        
        # Plot transition matrix
        print("\n🔄 Plotting transition matrix...")
        plot_state_transition_matrix(transition_matrix, output_dir)
        
        # Aging analysis
        print("\n🏗️ Plotting aging analysis...")
        data_dir = Path(__file__).parent / "data" / "private_benchmark"
        preprocessor = CBMDataPreprocessor(data_dir)
        preprocessor.load_data()
        plot_aging_analysis(preprocessor, args.equipment_id, args.measurement_id, output_dir)
        
        # QR-DQN Distribution Analysis
        if args.analyze_dist:
            print("\n📊 Performing QR-DQN distribution analysis...")
            analyze_qr_distribution(env, policy_net, output_dir, 
                                   n_samples=args.n_samples, device=str(device))
        
        # Evaluate policy
        print("\n🧪 Evaluating policy on test episodes...")
        episode_data = evaluate_policy(env, policy_net, n_episodes=10, device=str(device))
        
        avg_reward = np.mean([ep['total_reward'] for ep in episode_data])
        print(f"✅ Average reward: {avg_reward:.2f}")
        
        # Plot evaluation
        print("\n📊 Plotting policy evaluation...")
        plot_policy_evaluation(episode_data, output_dir)
        
    else:
        print(f"\n⚠️ Trained model not found: {model_path}")
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETED!")
    print("="*80)
    print("\nGenerated plots:")
    print("  • training_history.png - Comprehensive training curves")
    print("  • transition_matrix.png - State transition heatmap")
    print("  • aging_analysis.png - Equipment aging analysis")
    print("  • policy_evaluation.png - Test episode analysis")
    if args.analyze_dist:
        print("  • distribution_statistics.png - Return distribution shapes")
        print("  • uncertainty_analysis.png - Variance and IQR comparison")
        print("  • risk_profile.png - VaR and CVaR analysis")
        print("  • quantile_distributions.png - QR-DQN quantile functions")
    print("="*80)


if __name__ == "__main__":
    # Set matplotlib backend for non-interactive use
    import matplotlib
    matplotlib.use('Agg')
    
    main()
