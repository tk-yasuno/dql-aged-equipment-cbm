"""
実データから推定した遷移行列を使った環境テスト
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cbm_environment import EquipmentCBMEnvironment, STATE_NAMES, ACTION_NAMES
from data_preprocessor import CBMDataPreprocessor


def test_real_data_environment():
    """実データから推定した遷移行列で環境テスト"""
    
    print("="*60)
    print("🏭 Real Data CBM Environment Test")
    print("="*60)
    
    # データ読み込みと遷移行列推定
    data_dir = Path(__file__).parent.parent / "data" / "private_benchmark"
    preprocessor = CBMDataPreprocessor(data_dir)
    preprocessor.load_data()
    
    # ボイラー温度データ処理
    equipment_id = 43175
    measurement_id = 167473
    
    print(f"\n📊 Processing: 設備ID={equipment_id}, 測定項目ID={measurement_id}")
    stats = preprocessor.process_equipment(equipment_id, measurement_id)
    
    # 環境作成
    transition_matrix = np.array(stats['transition_matrix'], dtype=np.float32)
    temp_min = stats['value_stats']['min']
    temp_max = stats['value_stats']['max']
    normal_temp_min = stats['thresholds']['Smin']
    normal_temp_max = stats['thresholds']['Smax']
    
    env = EquipmentCBMEnvironment(
        transition_matrix=transition_matrix,
        temperature_range=(temp_min, temp_max),
        normal_temp_range=(normal_temp_min, normal_temp_max),
        horizon=50,
        seed=42,
        render_mode='human'
    )
    
    print("\n" + "="*60)
    print("✅ Environment created with REAL transition matrix")
    print("="*60)
    print(f"  - Temperature range: [{temp_min:.1f}, {temp_max:.1f}]°C")
    print(f"  - Normal range: [{normal_temp_min:.1f}, {normal_temp_max:.1f}]°C")
    print(f"  - Transition matrix (DoNothing):")
    print(f"    [[{transition_matrix[0,0]:.4f}, {transition_matrix[0,1]:.4f}],  # Normal → [Normal, Anomalous]")
    print(f"     [{transition_matrix[1,0]:.4f}, {transition_matrix[1,1]:.4f}]]  # Anomalous → [Normal, Anomalous]")
    
    # テストエピソード実行
    print("\n" + "="*60)
    print("🎬 Test Episode with Greedy Repair Policy")
    print("="*60)
    
    obs, info = env.reset(seed=42)
    print(f"\nInitial: condition={info['condition']}, temp={info['temperature']:.1f}°C\n")
    
    total_reward = 0.0
    actions_taken = []
    state_history = []
    
    for step in range(50):
        # 簡易方策: Anomalousなら修理、Normalなら何もしない
        if env.current_condition == 1:  # Anomalous
            # 重度ならReplace、軽度ならRepair
            if env.current_temperature > normal_temp_max + 20:
                action = 2  # Replace
            else:
                action = 1  # Repair
        else:  # Normal
            action = 0  # DoNothing
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        actions_taken.append(info['action'])
        state_history.append(info['old_condition'])
        
        print(f"  Step {step+1:2d} | Action: {info['action']:10s} | "
              f"{info['old_condition']:10s} → {info['new_condition']:10s} | "
              f"Reward: {reward:6.2f} | Temp: {info['temperature']:5.1f}°C")
        
        if terminated or truncated:
            break
    
    # サマリー
    print("\n" + "="*60)
    print("📊 Episode Summary")
    print("="*60)
    print(f"  - Total steps: {env.current_step}")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Average reward per step: {total_reward/env.current_step:.2f}")
    
    # 行動分布
    action_dist = {a: actions_taken.count(a) for a in set(actions_taken)}
    print(f"  - Actions taken: {action_dist}")
    
    # 状態分布
    state_dist = {STATE_NAMES[s]: state_history.count(s) for s in set(state_history)}
    print(f"  - States visited: {state_dist}")
    
    # 状態推移統計
    normal_count = state_history.count(0)
    anomalous_count = state_history.count(1)
    print(f"  - State distribution: Normal {normal_count}/{len(state_history)} ({normal_count/len(state_history)*100:.1f}%), "
          f"Anomalous {anomalous_count}/{len(state_history)} ({anomalous_count/len(state_history)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ Real Data Environment Test Completed!")
    print("="*60)


if __name__ == "__main__":
    test_real_data_environment()
