"""
Visualize existing scenario comparison results
既存のシナリオ訓練結果を可視化
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from pathlib import Path

# Set Japanese font
try:
    # Windows
    rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
except:
    pass
rcParams['axes.unicode_minus'] = False

SCENARIOS = {
    'safety_first': {
        'name': '安全重視',
        'description': '設備停止を回避し、積極的に保全',
        'output_dir': 'outputs_safety_first',
        'color': '#FF6B6B'
    },
    'balanced': {
        'name': 'バランス型',
        'description': '安全とコストを両立した保全戦略',
        'output_dir': 'outputs_balanced',
        'color': '#4ECDC4'
    },
    'cost_efficient': {
        'name': 'コスト重視',
        'description': '設備中断を許容し、必要最小限の保全',
        'output_dir': 'outputs_cost_efficient',
        'color': '#95E1D3'
    }
}


def load_training_history(output_dir: str):
    """Load training history from JSON file"""
    history_file = Path(output_dir) / "training_history.json"
    
    if not history_file.exists():
        print(f"❌ Training history not found: {history_file}")
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    return history


def visualize_individual_scenarios():
    """Create detailed visualization for each scenario"""
    
    for scenario_key, scenario_info in SCENARIOS.items():
        output_dir = scenario_info['output_dir']
        history = load_training_history(output_dir)
        
        if history is None:
            print(f"⚠️ Skipping {scenario_info['name']} - no training data")
            continue
        
        rewards = np.array(history['episode_rewards'])
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"{scenario_info['name']} - Detailed Analysis", 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # 1. Raw rewards
        ax1 = axes[0, 0]
        ax1.plot(rewards, alpha=0.3, color=scenario_info['color'], linewidth=0.5)
        ax1.set_xlabel('Episode', fontsize=11)
        ax1.set_ylabel('Episode Reward', fontsize=11)
        ax1.set_title('Raw Episode Rewards', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # 2. Moving average (window=50)
        ax2 = axes[0, 1]
        window = 50
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax2.plot(moving_avg, color=scenario_info['color'], linewidth=2)
        ax2.set_xlabel('Episode', fontsize=11)
        ax2.set_ylabel('Moving Average Reward', fontsize=11)
        ax2.set_title(f'Moving Average (window={window})', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # 3. Distribution histogram
        ax3 = axes[1, 0]
        ax3.hist(rewards, bins=50, color=scenario_info['color'], alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(rewards):.2f}')
        ax3.axvline(np.median(rewards), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(rewards):.2f}')
        ax3.set_xlabel('Reward', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Reward Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Cumulative average
        ax4 = axes[1, 1]
        cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        ax4.plot(cumulative_avg, color=scenario_info['color'], linewidth=2)
        ax4.set_xlabel('Episode', fontsize=11)
        ax4.set_ylabel('Cumulative Average Reward', fontsize=11)
        ax4.set_title('Cumulative Average Reward', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add statistics text
        stats_text = f"""Statistics:
Mean: {np.mean(rewards):.2f}
Std: {np.std(rewards):.2f}
Min: {np.min(rewards):.2f}
Max: {np.max(rewards):.2f}
Final 100 avg: {np.mean(rewards[-100:]):.2f}"""
        
        ax4.text(0.98, 0.02, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path('outputs_comparison') / f'{scenario_key}_detailed.png'
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved detailed visualization: {output_path}")


def compare_results():
    """Create comparison plots"""
    
    # Load all scenarios
    scenario_data = {}
    for scenario_key, scenario_info in SCENARIOS.items():
        history = load_training_history(scenario_info['output_dir'])
        if history is not None:
            scenario_data[scenario_key] = {
                'rewards': np.array(history['episode_rewards']),
                'info': scenario_info
            }
    
    if not scenario_data:
        print("❌ No training data found for any scenario")
        return
    
    # Create comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Scenario Comparison: Learning Curves', fontsize=16, fontweight='bold')
    
    # Plot learning curves
    for scenario_key, data in scenario_data.items():
        rewards = data['rewards']
        info = data['info']
        
        # Moving average
        window = 50
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(moving_avg, label=info['name'], color=info['color'], linewidth=2)
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Moving Average Reward (window=50)', fontsize=12)
    ax1.set_title('Learning Curves Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Bar chart for final performance
    scenario_names = [data['info']['name'] for data in scenario_data.values()]
    final_rewards = [np.mean(data['rewards'][-100:]) for data in scenario_data.values()]
    colors = [data['info']['color'] for data in scenario_data.values()]
    
    bars = ax2.bar(scenario_names, final_rewards, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Final 100-Episode Average Reward', fontsize=12)
    ax2.set_title('Scenario Comparison: Final Performance', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, final_rewards):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save comparison
    output_path = Path('outputs_comparison') / 'scenario_comparison.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comparison plot: {output_path}")
    
    # Print statistics table
    print("\n" + "="*80)
    print("📊 SCENARIO COMPARISON RESULTS")
    print("="*80)
    print(f"{'Scenario':<15} {'Mean':<10} {'Final 100':<12} {'Max':<10} {'Std Dev':<10}")
    print("-"*80)
    
    for scenario_key, data in scenario_data.items():
        rewards = data['rewards']
        info = data['info']
        print(f"{info['name']:<15} "
              f"{np.mean(rewards):<10.2f} "
              f"{np.mean(rewards[-100:]):<12.2f} "
              f"{np.max(rewards):<10.2f} "
              f"{np.std(rewards):<10.2f}")
    
    print("="*80)
    
    # Find best scenario
    best_scenario = max(scenario_data.items(), 
                       key=lambda x: np.mean(x[1]['rewards'][-100:]))
    print(f"\n🏆 推奨シナリオ: {best_scenario[1]['info']['name']} "
          f"(最終100エピソード平均: {np.mean(best_scenario[1]['rewards'][-100:]):.2f})")
    print("="*80 + "\n")


def main():
    """Main function"""
    print("\n" + "="*80)
    print("📊 MAINTENANCE SCENARIO VISUALIZATION")
    print("="*80)
    print("Visualizing existing training results for all scenarios\n")
    
    # Create individual detailed plots
    print("Creating detailed visualizations for each scenario...")
    visualize_individual_scenarios()
    
    print("\nCreating comparison plots...")
    compare_results()
    
    print("\n✅ All visualizations completed!")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    matplotlib.use('Agg')
    import sys
    sys.exit(main())
