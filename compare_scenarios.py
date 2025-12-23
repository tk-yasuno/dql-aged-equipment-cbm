"""
Compare Different Maintenance Scenarios

Run training for 3 different maintenance scenarios and compare results:
1. safety_first: 安全重視 - 設備停止を回避し、積極的に保全
2. cost_efficient: コスト重視 - 設備中断を許容し、必要最小限の保全  
3. balanced: バランス型 - 安全とコストを両立した保全戦略
"""

import subprocess
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

# Set Japanese font
try:
    # Windows
    rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
except:
    pass
rcParams['axes.unicode_minus'] = False

# Training configuration
EPISODES = 1000
N_ENVS = 16

SCENARIOS = {
    'safety_first': {
        'name': '安全重視',
        'description': '設備停止を回避し、積極的に保全',
        'output_dir': 'outputs_safety_first'
    },
    'cost_efficient': {
        'name': 'コスト重視',
        'description': '設備中断を許容し、必要最小限の保全',
        'output_dir': 'outputs_cost_efficient'
    },
    'balanced': {
        'name': 'バランス型',
        'description': '安全とコストを両立した保全戦略',
        'output_dir': 'outputs_balanced'
    }
}

def run_scenario(scenario_key: str, python_exe: str):
    """Run training for a specific scenario"""
    scenario = SCENARIOS[scenario_key]
    print(f"\n{'='*80}")
    print(f"🚀 Running Scenario: {scenario['name']}")
    print(f"   {scenario['description']}")
    print(f"{'='*80}\n")
    
    cmd = [
        python_exe,
        'equipment-cbm-mvp/train_cbm_dqn_v2.py',
        '--episodes', str(EPISODES),
        '--n_envs', str(N_ENVS),
        '--scenario', scenario_key,
        '--output_dir', scenario['output_dir']
    ]
    
    result = subprocess.run(cmd, cwd=Path.cwd())
    
    if result.returncode != 0:
        print(f"❌ Training failed for {scenario['name']}")
        return False
    
    print(f"\n✅ Training completed for {scenario['name']}\n")
    return True


def visualize_individual_scenarios(results):
    """Visualize detailed training curves for each scenario"""
    
    print("\n📊 Generating individual scenario visualizations...\n")
    
    comparison_dir = Path('outputs_comparison')
    comparison_dir.mkdir(exist_ok=True)
    
    for scenario_key in ['safety_first', 'balanced', 'cost_efficient']:
        if scenario_key not in results:
            continue
        
        r = results[scenario_key]
        rewards = r['rewards']
        
        # Create detailed plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Raw rewards
        ax = axes[0, 0]
        ax.plot(rewards, alpha=0.4, color='blue')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title(f"{r['name']}: Episode Rewards\n{r['description']}")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 2. Moving average
        ax = axes[0, 1]
        window = 50
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), ma, color='darkblue', linewidth=2)
            ax.fill_between(range(window-1, len(rewards)), ma, alpha=0.3)
        ax.set_xlabel('Episode')
        ax.set_ylabel(f'Reward (MA {window})')
        ax.set_title('Moving Average Reward')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 3. Reward distribution
        ax = axes[1, 0]
        ax.hist(rewards, bins=50, alpha=0.7, edgecolor='black', color='green')
        ax.axvline(r['mean_reward'], color='red', linestyle='--', linewidth=2, 
                  label=f"Mean: {r['mean_reward']:.2f}")
        ax.axvline(r['final_100'], color='orange', linestyle='--', linewidth=2,
                  label=f"Final 100: {r['final_100']:.2f}")
        ax.set_xlabel('Total Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Learning progress (cumulative)
        ax = axes[1, 1]
        cumulative = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        ax.plot(cumulative, color='purple', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Average Reward')
        ax.set_title('Learning Progress (Cumulative Average)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save
        plot_path = comparison_dir / f"{scenario_key}_detailed.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ {r['name']}: {plot_path}")
        plt.close()
    
    print("\n✅ Individual scenario visualizations completed\n")


def compare_results():
    """Load and compare results from all scenarios"""
    
    print(f"\n{'='*80}")
    print("📊 COMPARING SCENARIOS")
    print(f"{'='*80}\n")
    
    results = {}
    
    for scenario_key, scenario in SCENARIOS.items():
        output_dir = Path(scenario['output_dir'])
        history_path = output_dir / 'training_history.json'
        
        if not history_path.exists():
            print(f"⚠️ No results found for {scenario['name']}")
            continue
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        rewards = history['episode_rewards']
        
        results[scenario_key] = {
            'name': scenario['name'],
            'description': scenario['description'],
            'rewards': rewards,
            'mean_reward': np.mean(rewards),
            'final_100': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards)
        }
    
    # Print comparison table
    print("シナリオ別性能比較:")
    print("-" * 80)
    print(f"{'シナリオ':<15} {'平均報酬':<12} {'最終100平均':<12} {'最大報酬':<12} {'標準偏差':<12}")
    print("-" * 80)
    
    for scenario_key in ['safety_first', 'balanced', 'cost_efficient']:
        if scenario_key in results:
            r = results[scenario_key]
            print(f"{r['name']:<15} {r['mean_reward']:>10.2f}  {r['final_100']:>10.2f}  "
                  f"{r['max_reward']:>10.2f}  {r['std_reward']:>10.2f}")
    
    print("-" * 80)
    
    # Visualize individual scenarios
    visualize_individual_scenarios(results)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Learning curves
    colors = {'safety_first': 'blue', 'cost_efficient': 'red', 'balanced': 'green'}
    
    for scenario_key in ['safety_first', 'balanced', 'cost_efficient']:
        if scenario_key in results:
            r = results[scenario_key]
            rewards = r['rewards']
            
            # Moving average
            window = 50
            if len(rewards) >= window:
                ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(rewards)), ma, 
                        label=r['name'], color=colors.get(scenario_key, 'gray'), linewidth=2)
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward (MA 50)', fontsize=12)
    ax1.set_title('Scenario Comparison: Learning Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Final performance comparison
    scenario_names = []
    final_rewards = []
    
    for scenario_key in ['safety_first', 'balanced', 'cost_efficient']:
        if scenario_key in results:
            r = results[scenario_key]
            scenario_names.append(r['name'])
            final_rewards.append(r['final_100'])
    
    bars = ax2.bar(scenario_names, final_rewards, 
                   color=[colors.get(k, 'gray') for k in ['safety_first', 'balanced', 'cost_efficient'] if k in results],
                   alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Average Reward (Last 100 episodes)', fontsize=12)
    ax2.set_title('Scenario Comparison: Final Performance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_dir = Path('outputs_comparison')
    comparison_dir.mkdir(exist_ok=True)
    plot_path = comparison_dir / 'scenario_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Comparison plot saved: {plot_path}")
    
    plt.show()
    
    print(f"\n{'='*80}")
    print("✅ COMPARISON COMPLETE")
    print(f"{'='*80}\n")
    
    # Recommendations
    print("💡 推奨シナリオ:")
    best_reward = max([r['final_100'] for r in results.values()])
    for scenario_key, r in results.items():
        if r['final_100'] == best_reward:
            print(f"   {r['name']}: {r['description']}")
            print(f"   最終平均報酬: {r['final_100']:.2f}")
            break


def main():
    python_exe = sys.executable
    
    print("\n" + "="*80)
    print("🔬 MAINTENANCE SCENARIO COMPARISON")
    print("="*80)
    print(f"Training {EPISODES} episodes for each scenario with {N_ENVS} parallel environments\n")
    
    for key in ['safety_first', 'balanced', 'cost_efficient']:
        print(f"{SCENARIOS[key]['name']:<15}: {SCENARIOS[key]['description']}")
    
    print("="*80)
    
    # Run all scenarios
    success = True
    for scenario_key in ['safety_first', 'balanced', 'cost_efficient']:
        if not run_scenario(scenario_key, python_exe):
            success = False
            break
    
    if success:
        # Compare results
        compare_results()
    else:
        print("\n❌ Training failed for one or more scenarios")
        return 1
    
    return 0


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    sys.exit(main())
