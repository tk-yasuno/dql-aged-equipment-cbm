"""
6台設備CBM強化学習結果比較分析スクリプト

各設備の特性に応じた政策の違いを定量的に比較分析
- 学習収束性能
- リスク回避性
- 政策評価結果
- 設備年数と学習戦略の関係
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Tuple

# 日本語フォント設定
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Windowsで日本語表示可能なフォントを設定
def setup_japanese_font():
    """日本語フォントを設定する"""
    # Windows標準の日本語フォント候補
    japanese_fonts = [
        'Yu Gothic UI',  # Windows 10/11標準
        'Yu Gothic', 
        'Meiryo UI', 
        'Meiryo',
        'MS Gothic',
        'MS UI Gothic'
    ]
    
    # システムで利用可能なフォント一覧を取得
    system_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 利用可能な日本語フォントを探す
    for font in japanese_fonts:
        if font in system_fonts:
            plt.rcParams['font.family'] = font
            print(f"✓ 日本語フォントを設定: {font}")
            return font
    
    # 日本語フォントが見つからない場合のフォールバック
    print("⚠️ 日本語フォントが見つかりません。デフォルトフォントを使用します。")
    return None

# フォント設定を実行
setup_japanese_font()
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False  # マイナス符号の文字化け防止

# 設備情報定義
EQUIPMENT_INFO = {
    'outputs_pump_265715': {
        'name': '薬注ポンプCP-500-5',
        'age': 19.7,
        'category': '老朽化設備',
        'equipment_id': 265715,
        'aging_factor': 0.018
    },
    'outputs_pump_137953': {
        'name': '冷却水ポンプCDP-A5',
        'age': 3.0,
        'category': '新しい設備',
        'equipment_id': 137953,
        'aging_factor': 0.005
    },
    'outputs_pump_519177': {
        'name': '薬注ポンプCP-500-3',
        'age': 0.5,
        'category': '最新設備',
        'equipment_id': 519177,
        'aging_factor': 0.003
    },
    'outputs_ahu_327240': {
        'name': 'AHU-TSK-A-2',
        'age': 15.6,
        'category': 'エアハンドリングユニット',
        'equipment_id': 327240,
        'aging_factor': 0.015
    },
    'outputs_r13_265694': {
        'name': 'R-1-3',
        'age': 19.7,
        'category': '冷却器設備',
        'equipment_id': 265694,
        'aging_factor': 0.018
    },
    'outputs_oac_322220': {
        'name': 'OAC-TSK-F-2',
        'age': 17.7,
        'category': '外気処理機',
        'equipment_id': 322220,
        'aging_factor': 0.015
    },
    'outputs_ahu_327280_dp1': {
        'name': 'AHU-TSK-F-4',
        'age': 14.2,
        'category': 'エアハンドリングユニット',
        'equipment_id': 327280,
        'aging_factor': 0.0151
    }
}

def load_training_history(output_dir: Path) -> Dict:
    """学習履歴を読み込み"""
    history_path = output_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            return json.load(f)
    return None

def analyze_training_performance(histories: Dict) -> pd.DataFrame:
    """学習性能を分析"""
    results = []
    
    for output_dir, history in histories.items():
        if history is None:
            continue
            
        info = EQUIPMENT_INFO[output_dir]
        rewards = history['episode_rewards']
        
        # 基本統計
        final_100_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        final_performance = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
        convergence_stability = np.std(rewards[-100:]) if len(rewards) >= 100 else np.std(rewards)
        
        # 収束速度（平均報酬が一定値を超えた最初のエピソード）
        threshold = np.mean(rewards[-100:]) * 0.9 if len(rewards) >= 100 else np.mean(rewards) * 0.9
        convergence_episode = None
        for i, reward in enumerate(rewards):
            if i >= 100:  # 最初の100エピソードは除く
                window_avg = np.mean(rewards[max(0, i-50):i+1])
                if window_avg >= threshold:
                    convergence_episode = i
                    break
        
        results.append({
            'Equipment': info['name'],
            'Age (years)': info['age'],
            'Category': info['category'],
            'Aging Factor': info['aging_factor'],
            'Final Reward (100ep avg)': final_100_avg,
            'Final Performance (50ep avg)': final_performance,
            'Stability (std)': convergence_stability,
            'Convergence Episode': convergence_episode or len(rewards),
            'Total Episodes': len(rewards),
            'Max Reward': np.max(rewards),
            'Min Reward': np.min(rewards)
        })
    
    return pd.DataFrame(results)

def plot_learning_curves_comparison(histories: Dict, save_path: Path):
    """学習カーブの比較プロット"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(histories)))
    
    for i, (output_dir, history) in enumerate(histories.items()):
        if history is None:
            continue
            
        info = EQUIPMENT_INFO[output_dir]
        rewards = history['episode_rewards']
        color = colors[i]
        
        # 移動平均
        window = 50
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            episodes = np.arange(window-1, len(rewards))
            ax1.plot(episodes, smoothed, label=f"{info['name']} ({info['age']}年)", 
                    color=color, linewidth=2)
        
        # Raw rewards (thin lines)
        ax2.plot(rewards, alpha=0.6, color=color, linewidth=1)
    
    ax1.set_title('Learning Curves Comparison (50-Episode Moving Average)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Raw Reward Values Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.grid(True, alpha=0.3)
    
    # 設備年数と最終性能の関係
    ages = [EQUIPMENT_INFO[dir_name]['age'] for dir_name in histories.keys()]
    final_rewards = []
    
    for output_dir, history in histories.items():
        if history is None:
            final_rewards.append(0)
            continue
        rewards = history['episode_rewards']
        final_rewards.append(np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards))
    
    scatter = ax3.scatter(ages, final_rewards, c=colors, s=100, alpha=0.7)
    ax3.set_title('Equipment Age vs Final Performance', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Equipment Age (years)')
    ax3.set_ylabel('Final Reward (last 100 episodes avg)')
    ax3.grid(True, alpha=0.3)
    
    # 注釈を追加
    for i, (age, reward) in enumerate(zip(ages, final_rewards)):
        output_dir = list(histories.keys())[i]
        info = EQUIPMENT_INFO[output_dir]
        ax3.annotate(info['name'].split('-')[0], (age, reward), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 老朽化係数と最終性能の関係
    aging_factors = [EQUIPMENT_INFO[dir_name]['aging_factor'] for dir_name in histories.keys()]
    ax4.scatter(aging_factors, final_rewards, c=colors, s=100, alpha=0.7)
    ax4.set_title('Aging Factor vs Final Performance', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Aging Factor')
    ax4.set_ylabel('Final Reward (last 100 episodes avg)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / "learning_curves_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_analysis(df: pd.DataFrame, save_path: Path):
    """性能分析の詳細プロット"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 設備カテゴリ別最終性能
    category_performance = df.groupby('Category')['Final Reward (100ep avg)'].agg(['mean', 'std'])
    ax1.bar(range(len(category_performance)), category_performance['mean'], 
            yerr=category_performance['std'], capsize=5, alpha=0.7)
    ax1.set_xticks(range(len(category_performance)))
    ax1.set_xticklabels(category_performance.index, rotation=45, ha='right')
    ax1.set_title('Final Performance by Equipment Category', fontweight='bold')
    ax1.set_ylabel('Average Reward')
    ax1.grid(True, alpha=0.3)
    
    # 2. 年数vs性能の散布図（回帰線付き）
    ax2.scatter(df['Age (years)'], df['Final Reward (100ep avg)'], s=100, alpha=0.7)
    
    # 回帰線
    z = np.polyfit(df['Age (years)'], df['Final Reward (100ep avg)'], 1)
    p = np.poly1d(z)
    ax2.plot(df['Age (years)'], p(df['Age (years)']), "r--", alpha=0.8)
    
    # 相関係数
    correlation = np.corrcoef(df['Age (years)'], df['Final Reward (100ep avg)'])[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax2.set_title('Equipment Age vs Final Performance', fontweight='bold')
    ax2.set_xlabel('Equipment Age (years)')
    ax2.set_ylabel('Final Reward')
    ax2.grid(True, alpha=0.3)
    
    # 3. 収束安定性比較
    stability_sorted = df.sort_values('Stability (std)')
    ax3.barh(range(len(stability_sorted)), stability_sorted['Stability (std)'], alpha=0.7)
    ax3.set_yticks(range(len(stability_sorted)))
    ax3.set_yticklabels([name.split('-')[0] for name in stability_sorted['Equipment']])
    ax3.set_title('Convergence Stability (Std Dev)', fontweight='bold')
    ax3.set_xlabel('Reward Standard Deviation')
    ax3.grid(True, alpha=0.3)
    
    # 4. 収束速度比較
    convergence_sorted = df.sort_values('Convergence Episode')
    ax4.barh(range(len(convergence_sorted)), convergence_sorted['Convergence Episode'], alpha=0.7)
    ax4.set_yticks(range(len(convergence_sorted)))
    ax4.set_yticklabels([name.split('-')[0] for name in convergence_sorted['Equipment']])
    ax4.set_title('Convergence Speed (Episodes to Converge)', fontweight='bold')
    ax4.set_xlabel('Episodes to Convergence')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / "performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_report(df: pd.DataFrame, save_path: Path):
    """比較分析レポートを生成"""
    report_path = save_path / "equipment_comparison_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 6台設備CBM強化学習結果 比較分析レポート\n\n")
        f.write(f"生成日時: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
        
        f.write("## 1. 実行概要\n\n")
        f.write("6つの異なる年数・タイプの設備について、QR-DQN強化学習を用いて\n")
        f.write("条件ベースメンテナンス(CBM)政策を学習し、その結果を比較分析しました。\n\n")
        
        f.write("## 2. 設備一覧\n\n")
        f.write("| 設備名 | 年数 | カテゴリ | 老朽化係数 | 最終性能 |\n")
        f.write("|--------|------|----------|------------|----------|\n")
        
        for _, row in df.iterrows():
            f.write(f"| {row['Equipment']} | {row['Age (years)']}年 | {row['Category']} | "
                   f"{row['Aging Factor']:.3f} | {row['Final Reward (100ep avg)']:.2f} |\n")
        
        f.write("\n## 3. 主要な発見事項\n\n")
        
        # 最高性能と最低性能
        best_performer = df.loc[df['Final Reward (100ep avg)'].idxmax()]
        worst_performer = df.loc[df['Final Reward (100ep avg)'].idxmin()]
        
        f.write(f"### 3.1 性能分析\n")
        f.write(f"- **最高性能設備**: {best_performer['Equipment']} (報酬: {best_performer['Final Reward (100ep avg)']:.2f})\n")
        f.write(f"- **最低性能設備**: {worst_performer['Equipment']} (報酬: {worst_performer['Final Reward (100ep avg)']:.2f})\n")
        f.write(f"- **性能差**: {best_performer['Final Reward (100ep avg)'] - worst_performer['Final Reward (100ep avg)']:.2f}\n\n")
        
        # 年数と性能の関係
        age_correlation = np.corrcoef(df['Age (years)'], df['Final Reward (100ep avg)'])[0, 1]
        f.write(f"### 3.2 設備年数の影響\n")
        f.write(f"- **年数vs性能相関**: {age_correlation:.3f}\n")
        if age_correlation < -0.3:
            f.write("- 設備の老朽化に伴い学習性能が低下する傾向が確認されました\n")
        elif age_correlation > 0.3:
            f.write("- 興味深いことに、古い設備の方が高い性能を示しています\n")
        else:
            f.write("- 設備年数と性能には明確な相関は見られません\n")
        f.write("\n")
        
        # 収束特性
        fastest_convergence = df.loc[df['Convergence Episode'].idxmin()]
        slowest_convergence = df.loc[df['Convergence Episode'].idxmax()]
        most_stable = df.loc[df['Stability (std)'].idxmin()]
        
        f.write(f"### 3.3 学習特性\n")
        f.write(f"- **最速収束**: {fastest_convergence['Equipment']} ({fastest_convergence['Convergence Episode']}エピソード)\n")
        f.write(f"- **最遅収束**: {slowest_convergence['Equipment']} ({slowest_convergence['Convergence Episode']}エピソード)\n")
        f.write(f"- **最安定**: {most_stable['Equipment']} (標準偏差: {most_stable['Stability (std)']:.3f})\n\n")
        
        f.write("## 4. 設備カテゴリ別分析\n\n")
        category_stats = df.groupby('Category').agg({
            'Final Reward (100ep avg)': ['mean', 'std', 'count'],
            'Age (years)': 'mean'
        }).round(2)
        
        f.write("| カテゴリ | 平均性能 | 性能標準偏差 | 設備数 | 平均年数 |\n")
        f.write("|----------|----------|--------------|---------|----------|\n")
        
        for category in category_stats.index:
            f.write(f"| {category} | "
                   f"{category_stats.loc[category, ('Final Reward (100ep avg)', 'mean')]} | "
                   f"{category_stats.loc[category, ('Final Reward (100ep avg)', 'std')]} | "
                   f"{int(category_stats.loc[category, ('Final Reward (100ep avg)', 'count')])} | "
                   f"{category_stats.loc[category, ('Age (years)', 'mean')]} |\n")
        
        f.write("\n## 5. 政策学習の示唆\n\n")
        f.write("### 5.1 老朽化設備への対応\n")
        old_equipment = df[df['Age (years)'] > 15]
        if not old_equipment.empty:
            avg_old_performance = old_equipment['Final Reward (100ep avg)'].mean()
            f.write(f"- 15年以上の老朽化設備の平均性能: {avg_old_performance:.2f}\n")
            f.write("- 老朽化設備では予防保全の重要性が増加\n")
        
        f.write("\n### 5.2 新しい設備の特性\n")
        new_equipment = df[df['Age (years)'] < 5]
        if not new_equipment.empty:
            avg_new_performance = new_equipment['Final Reward (100ep avg)'].mean()
            f.write(f"- 5年未満の新しい設備の平均性能: {avg_new_performance:.2f}\n")
            f.write("- 新しい設備では効率的な運用が可能\n")
        
        f.write("\n## 6. 生成ファイル\n\n")
        f.write("以下の可視化ファイルが生成されました:\n")
        f.write("- `learning_curves_comparison.png` - 学習カーブ比較\n")
        f.write("- `performance_analysis.png` - 性能分析詳細\n")
        f.write("- `equipment_comparison_report.md` - 本レポート\n")
        f.write("- `comparison_summary.csv` - 数値データサマリー\n\n")
        
        f.write("各設備のディレクトリには個別の詳細分析結果も保存されています。\n")

def main():
    """メイン実行関数"""
    print("="*80)
    print("🔍 6台設備CBM強化学習結果 比較分析")
    print("="*80)
    
    # 学習履歴の読み込み
    histories = {}
    for output_dir in EQUIPMENT_INFO.keys():
        path = Path(output_dir)
        if path.exists():
            print(f"📊 {output_dir} の学習履歴を読み込み中...")
            histories[output_dir] = load_training_history(path)
        else:
            print(f"⚠️ {output_dir} が見つかりません")
    
    if not histories:
        print("❌ 有効な学習履歴が見つかりませんでした")
        return
    
    print(f"✅ {len(histories)}台の設備データを読み込み完了")
    
    # 結果保存ディレクトリ
    comparison_dir = Path("comparison_analysis")
    comparison_dir.mkdir(exist_ok=True)
    
    # 性能分析
    print("\n📈 学習性能を分析中...")
    performance_df = analyze_training_performance(histories)
    
    # CSV保存
    performance_df.to_csv(comparison_dir / "comparison_summary.csv", 
                         index=False, encoding='utf-8')
    
    # 可視化
    print("\n📊 比較グラフを生成中...")
    plot_learning_curves_comparison(histories, comparison_dir)
    plot_performance_analysis(performance_df, comparison_dir)
    
    # レポート生成
    print("\n📝 比較レポートを生成中...")
    generate_comparison_report(performance_df, comparison_dir)
    
    print("\n" + "="*80)
    print("✅ 比較分析完了！")
    print("="*80)
    print(f"📁 結果保存先: {comparison_dir.absolute()}")
    print("\n生成ファイル:")
    print("  • learning_curves_comparison.png - 学習カーブ比較")
    print("  • performance_analysis.png - 詳細性能分析")
    print("  • equipment_comparison_report.md - 比較分析レポート")
    print("  • comparison_summary.csv - 数値サマリー")
    print("="*80)
    
    # 結果の簡易表示
    print("\n📋 結果サマリー:")
    print(performance_df[['Equipment', 'Age (years)', 'Final Reward (100ep avg)', 'Convergence Episode']].to_string(index=False))

if __name__ == "__main__":
    # 非インタラクティブ使用のためmatplotlib設定
    import matplotlib
    matplotlib.use('Agg')
    
    main()