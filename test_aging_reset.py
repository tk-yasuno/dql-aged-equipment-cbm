"""
設備交換時の経過年数リセット機能のテスト

Replace行動実行時に設備年数がゼロにリセットされるかを確認
"""

import numpy as np
from pathlib import Path
import sys

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))
from cbm_environment import EquipmentCBMEnvironment, ACTION_NAMES

def test_equipment_age_reset():
    """設備年数リセット機能のテスト"""
    
    print("="*60)
    print("🧪 設備年数リセット機能テスト")
    print("="*60)
    
    # サンプル遷移行列
    transition_matrix = np.array([
        [0.95, 0.05],  # Normal → [Normal, Anomalous]
        [0.10, 0.90],  # Anomalous → [Normal, Anomalous]
    ], dtype=np.float32)
    
    # 環境を作成（初期年数を10年に設定）
    initial_age = 10.0
    aging_factor = 0.02  # 年間2%の異常率増加
    
    env = EquipmentCBMEnvironment(
        transition_matrix=transition_matrix,
        temperature_range=(0.0, 150.0),
        normal_temp_range=(20.0, 100.0),
        horizon=100,
        gamma=0.95,
        equipment_age=initial_age,
        aging_factor=aging_factor,
        seed=42
    )
    
    print(f"📅 初期設備年数: {initial_age} 年")
    print(f"📈 老朽化係数: {aging_factor} /年")
    
    # 初期リセット
    obs, info = env.reset()
    print(f"\nリセット後:")
    print(f"  観測値: {obs}")
    print(f"  設備年数: {info['equipment_age']:.2f} 年")
    print(f"  状態: {info['condition']}")
    print(f"  温度: {info['temperature']:.1f}°C")
    
    # 老朽化による遷移確率の変化を確認
    base_trans = env.transitions[0]  # DoNothing
    age_adjusted = env._get_age_adjusted_transition(base_trans)
    
    print(f"\n🔄 遷移確率比較:")
    print(f"  基本遷移行列:")
    print(f"    Normal→Normal: {base_trans[0,0]:.4f}")
    print(f"    Normal→Anomalous: {base_trans[0,1]:.4f}")
    print(f"  年数調整後 (年数 {env.equipment_age:.1f}):")
    print(f"    Normal→Normal: {age_adjusted[0,0]:.4f}")
    print(f"    Normal→Anomalous: {age_adjusted[0,1]:.4f}")
    print(f"  老朽化による異常率増加: +{(age_adjusted[0,1] - base_trans[0,1])*100:.2f}%")
    
    # 各アクションをテスト
    actions_to_test = [0, 1, 2]  # DoNothing, Repair, Replace
    
    for action in actions_to_test:
        print(f"\n" + "-"*50)
        print(f"🎯 アクション実行テスト: {ACTION_NAMES[action]} (ID: {action})")
        print("-"*50)
        
        # アクション前の状態
        age_before = env.equipment_age
        condition_before = env.current_condition
        
        print(f"実行前:")
        print(f"  設備年数: {age_before:.2f} 年")
        print(f"  状態: {condition_before} ({['Normal', 'Anomalous'][condition_before]})")
        
        # アクション実行
        obs, reward, terminated, truncated, info = env.step(action)
        
        age_after = env.equipment_age
        condition_after = env.current_condition
        
        print(f"実行後:")
        print(f"  設備年数: {age_after:.2f} 年")
        print(f"  状態: {condition_after} ({['Normal', 'Anomalous'][condition_after]})")
        print(f"  報酬: {reward:.2f}")
        print(f"  年数変化: {age_after - age_before:.2f} 年")
        
        # Replace実行時の特別チェック
        if action == 2:  # Replace
            if age_after == 0.0:
                print("  ✅ Replace実行時に設備年数が正常にリセットされました")
            else:
                print(f"  ❌ Replace実行時に設備年数がリセットされませんでした (期待値: 0.0, 実際: {age_after})")
        
        print(f"  観測値: {obs}")
    
    # 複数ステップでの年数進行をテスト
    print(f"\n" + "="*50)
    print("📊 複数ステップでの年数進行テスト")
    print("="*50)
    
    # 環境をリセット
    env.reset()
    
    # DoNothingを5回実行
    for step in range(5):
        age_before = env.equipment_age
        obs, reward, terminated, truncated, info = env.step(0)  # DoNothing
        age_after = env.equipment_age
        
        print(f"Step {step+1}: {age_before:.3f} → {age_after:.3f} 年 (変化: +{age_after-age_before:.3f})")
    
    # Replace実行
    print(f"\n🔄 Replace実行...")
    age_before = env.equipment_age
    obs, reward, terminated, truncated, info = env.step(2)  # Replace
    age_after = env.equipment_age
    
    print(f"Replace前: {age_before:.3f} 年")
    print(f"Replace後: {age_after:.3f} 年")
    
    if age_after == 0.0:
        print("✅ Replace実行により設備年数が正常にゼロリセットされました")
    else:
        print(f"❌ Replace実行後も設備年数が残っています: {age_after}")
    
    print(f"\n" + "="*60)
    print("🎉 テスト完了")
    print("="*60)

def test_aging_effect_on_transitions():
    """老朽化が状態遷移に与える影響のテスト"""
    
    print("\n" + "="*60)
    print("📈 老朽化効果の遷移確率テスト")
    print("="*60)
    
    base_transition = np.array([
        [0.95, 0.05],  # Normal → [Normal, Anomalous]
        [0.10, 0.90],  # Anomalous → [Normal, Anomalous]
    ], dtype=np.float32)
    
    aging_factor = 0.01  # 年間1%増加
    ages_to_test = [0, 5, 10, 15, 20, 25]
    
    print("設備年数別の異常への遷移確率:")
    print("年数\t基本確率\t調整後確率\t増加分")
    print("-" * 50)
    
    for age in ages_to_test:
        env = EquipmentCBMEnvironment(
            transition_matrix=base_transition,
            equipment_age=age,
            aging_factor=aging_factor,
            seed=42
        )
        
        adjusted = env._get_age_adjusted_transition(base_transition)
        base_prob = base_transition[0, 1]  # Normal→Anomalous
        adjusted_prob = adjusted[0, 1]
        increase = adjusted_prob - base_prob
        
        print(f"{age:2d}\t{base_prob:.4f}\t\t{adjusted_prob:.4f}\t\t+{increase:.4f}")

if __name__ == "__main__":
    test_equipment_age_reset()
    test_aging_effect_on_transitions()