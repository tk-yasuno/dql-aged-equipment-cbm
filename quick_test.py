"""簡易環境テスト - 実データ遷移行列"""
import numpy as np
from cbm_environment import EquipmentCBMEnvironment

# 実データから推定された遷移行列
transition_matrix = np.array([
    [0.2948, 0.7052],  # Normal → [Normal, Anomalous] 
    [0.0731, 0.9269]   # Anomalous → [Normal, Anomalous]
], dtype=np.float32)

print("="*60)
print("🏭 Real Data Environment Test")
print("="*60)

env = EquipmentCBMEnvironment(
    transition_matrix=transition_matrix,
    temperature_range=(11.5, 138.4),
    normal_temp_range=(13.02, 40.0),
    horizon=30,
    seed=42
)

print(f"\n✅ Environment created with real data")
print(f"   Transition probabilities:")
print(f"     Normal → Anomalous: 70.5%")
print(f"     Anomalous → Anomalous: 92.7%")
print(f"   Temperature range: [11.5, 138.4]°C")
print(f"   Normal range: [13.02, 40.0]°C")

print(f"\n🎬 Test Episode (30 steps)")
print(f"   Policy: Repair if Anomalous, DoNothing if Normal")
print("="*60)

obs, info = env.reset()
print(f"Initial: {info['condition']}, Temp={info['temperature']:.1f}°C\n")

total_reward = 0
actions = {'DoNothing': 0, 'Repair': 0, 'Replace': 0}
states = {'Normal': 0, 'Anomalous': 0}

for step in range(30):
    # Simple policy
    action = 1 if env.current_condition == 1 else 0
    
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    actions[info['action']] += 1
    states[info['old_condition']] += 1
    
    print(f"{step+1:2d}. {info['action']:10s} | "
          f"{info['old_condition']:10s} → {info['new_condition']:10s} | "
          f"R:{reward:6.2f} | T:{info['temperature']:5.1f}°C")
    
    if terminated or truncated:
        break

print("\n" + "="*60)
print("📊 Summary")
print("="*60)
print(f"Total reward: {total_reward:.2f}")
print(f"Avg reward/step: {total_reward/env.current_step:.2f}")
print(f"Actions: {actions}")
print(f"States: {states}")
print(f"State ratio: Normal {states['Normal']}/{env.current_step} ({states['Normal']/env.current_step*100:.1f}%)")
