"""
2x2 Markov CBM Environment for Equipment Maintenance

Features:
- 2x2 state transition: Normal / Anomalous
- Actions: DoNothing, Repair, Replace
- Reward: Risk suppression + Cost minimization
- Based on: base_markov-dqn-v09-quantile/src/markov_fleet_environment.py

State Definition:
- condition: 0=normal, 1=anomalous (based on CBM thresholds)
- temperature: normalized temperature value

Action Space:
- 0: Do Nothing (continue operation)
- 1: Repair (reset to normal, medium cost)
- 2: Replace (reset to normal, high cost)

Reward Function:
- Risk component: +1 for normal, -10 for anomalous
- Cost component: 0 for do nothing, -3 for repair, -8 for replace
"""

from typing import Optional, Tuple, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ----- Constants -----

STATE_NAMES = ["Normal", "Anomalous"]  # 0, 1
ACTION_NAMES = ["DoNothing", "Repair", "Replace"]  # 0, 1, 2

# Default 2x2 transition matrix (will be updated from real data)
DEFAULT_TRANSITIONS = {
    0: np.array([  # DoNothing
        [0.95, 0.05],  # from Normal → [Normal, Anomalous]
        [0.10, 0.90],  # from Anomalous → [Normal, Anomalous]
    ], dtype=np.float32),
    1: np.array([  # Repair
        [0.98, 0.02],  # from Normal → [Normal, Anomalous] (slightly improved)
        [0.80, 0.20],  # from Anomalous → [Normal, Anomalous] (high recovery)
    ], dtype=np.float32),
    2: np.array([  # Replace
        [0.99, 0.01],  # from Normal → [Normal, Anomalous] (best state)
        [0.95, 0.05],  # from Anomalous → [Normal, Anomalous] (almost full recovery)
    ], dtype=np.float32),
}

# Action costs (relative units)
ACTION_COSTS = np.array([
    0.0,   # DoNothing
    3.0,   # Repair
    15.0,  # Replace (increased for realism)
], dtype=np.float32)

# Maintenance Scenario Presets
MAINTENANCE_SCENARIOS = {
    'safety_first': {
        'risk_weight': 1.0,
        'cost_lambda': 0.05,  # Low cost penalty
        'description': '安全重視：設備停止を回避し、積極的に保全'
    },
    'cost_efficient': {
        'risk_weight': 0.3,
        'cost_lambda': 0.5,   # High cost penalty
        'description': 'コスト重視：設備中断を許容し、必要最小限の保全'
    },
    'balanced': {
        'risk_weight': 1.0,
        'cost_lambda': 0.15,  # Medium cost penalty
        'description': 'バランス型：安全とコストを両立した保全戦略'
    }
}


class EquipmentCBMEnvironment(gym.Env):
    """
    2x2 Markov CBM Environment for Equipment Maintenance
    
    State: [condition, normalized_temperature]
    - condition: 0=normal, 1=anomalous
    - normalized_temperature: 0.0~1.0 scaled from actual temperature
    
    Action: 0=DoNothing, 1=Repair, 2=Replace
    
    Reward: Risk suppression + Cost minimization
    """
    
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 1}
    
    def __init__(
        self,
        transition_matrix: Optional[np.ndarray] = None,
        temperature_range: Tuple[float, float] = (0.0, 150.0),
        normal_temp_range: Tuple[float, float] = (20.0, 100.0),
        horizon: int = 100,
        gamma: float = 0.95,
        risk_weight: float = 1.0,
        cost_lambda: float = 0.15,
        scenario: Optional[str] = None,
        equipment_age: float = 0.0,
        aging_factor: float = 0.01,  # 年間異常確率の增加率
        max_equipment_age: float = 50.0,  # 最大設備年数
        seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Args:
            transition_matrix: DoNothing時の2x2遷移行列 [[p_nn, p_na], [p_an, p_aa]]
            temperature_range: 温度の物理的範囲 (min, max)
            normal_temp_range: 正常範囲の温度 (min, max)
            horizon: エピソード長
            gamma: 割引率
            risk_weight: リスクペナルティの重み (大きいほど安全重視)
            cost_lambda: コストペナルティの重み (大きいほどコスト重視)
            scenario: プリセットシナリオ ('safety_first', 'cost_efficient', 'balanced')
            seed: 乱数シード
            render_mode: レンダリングモード
        """
        super().__init__()
        
        # Apply scenario preset if specified
        if scenario is not None:
            if scenario not in MAINTENANCE_SCENARIOS:
                raise ValueError(f"Unknown scenario: {scenario}. Choose from {list(MAINTENANCE_SCENARIOS.keys())}")
            preset = MAINTENANCE_SCENARIOS[scenario]
            risk_weight = preset['risk_weight']
            cost_lambda = preset['cost_lambda']
            print(f"\n📋 Maintenance Scenario: {scenario}")
            print(f"   {preset['description']}")
            print(f"   Risk Weight: {risk_weight}, Cost Lambda: {cost_lambda}\n")
        
        self.render_mode = render_mode
        self.horizon = horizon
        self.gamma = gamma
        self.risk_weight = risk_weight
        self.cost_lambda = cost_lambda
        self.scenario = scenario
        
        # Temperature settings
        self.temp_min, self.temp_max = temperature_range
        self.normal_temp_min, self.normal_temp_max = normal_temp_range
        
        # Equipment aging settings
        self.initial_equipment_age = equipment_age
        self.aging_factor = aging_factor  # 年間異常確率の增加率
        self.max_equipment_age = max_equipment_age
        self.equipment_age = equipment_age  # 現在の設備年数
        
        # Transition matrix
        if transition_matrix is not None:
            # Use provided transition matrix for DoNothing
            assert transition_matrix.shape == (2, 2), "Transition matrix must be 2x2"
            self.transitions = DEFAULT_TRANSITIONS.copy()
            self.transitions[0] = transition_matrix.astype(np.float32)
        else:
            self.transitions = DEFAULT_TRANSITIONS.copy()
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0=DoNothing, 1=Repair, 2=Replace
        
        # Observation: [condition (0 or 1), normalized_temperature (0~1), normalized_age (0~1)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.current_condition = 0  # 0=normal
        self.current_temperature = 0.0
        
        # Random seed
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()
    
    def _normalize_temperature(self, temp: float) -> float:
        """温度を0~1にスケール"""
        return (temp - self.temp_min) / (self.temp_max - self.temp_min)
    
    def _denormalize_temperature(self, norm_temp: float) -> float:
        """0~1スケールを実温度に戻す"""
        return norm_temp * (self.temp_max - self.temp_min) + self.temp_min
    
    def _sample_temperature(self, condition: int) -> float:
        """状態に応じた温度をサンプリング"""
        if condition == 0:  # Normal
            # 正常範囲内からサンプリング
            temp = self.np_random.uniform(self.normal_temp_min, self.normal_temp_max)
        else:  # Anomalous
            # 正常範囲外からサンプリング
            if self.np_random.rand() < 0.5:
                # 下限以下
                temp = self.np_random.uniform(self.temp_min, self.normal_temp_min)
            else:
                # 上限以上
                temp = self.np_random.uniform(self.normal_temp_max, self.temp_max)
        return temp
    
    def _get_age_adjusted_transition(self, base_transition: np.ndarray) -> np.ndarray:
        """設備年数を考慮した状態遷移行列を取得
        
        Args:
            base_transition: 基本の遷移行列
        
        Returns:
            老朽化を考慮した遷移行列
        """
        # 老朽化による異常への遷移確率増加
        aging_effect = self.equipment_age * self.aging_factor
        adjusted = base_transition.copy()
        
        # Normal状態からAnomalousへの遷移確率を増加
        if adjusted[0, 1] + aging_effect < 1.0:
            adjusted[0, 1] += aging_effect
            adjusted[0, 0] = 1.0 - adjusted[0, 1]
        else:
            # 上限に達した場合
            adjusted[0, 1] = 0.99
            adjusted[0, 0] = 0.01
            
        # Anomalous状態からNormalへの回復確率を若干減少
        recovery_penalty = aging_effect * 0.5  # 軽微な影響
        if adjusted[1, 0] - recovery_penalty > 0.0:
            adjusted[1, 0] -= recovery_penalty
            adjusted[1, 1] = 1.0 - adjusted[1, 0]
        
        return adjusted
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """環境をリセット"""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        # 初期状態: Normal
        self.current_step = 0
        self.current_condition = 0
        self.current_temperature = self._sample_temperature(0)
        self.equipment_age = self.initial_equipment_age  # 設備年数をリセット
        
        obs = self._get_observation()
        info = {
            'condition': STATE_NAMES[self.current_condition],
            'temperature': self.current_temperature,
            'equipment_age': self.equipment_age
        }
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """観測値を取得"""
        norm_temp = self._normalize_temperature(self.current_temperature)
        norm_age = min(self.equipment_age / self.max_equipment_age, 1.0)  # 0~1に正規化
        return np.array([float(self.current_condition), norm_temp, norm_age], dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """環境を1ステップ進める
        
        Args:
            action: 0=DoNothing, 1=Repair, 2=Replace
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Current state
        old_condition = self.current_condition
        
        # --- Risk Reward Component ---
        # Normal: +1, Anomalous: -10 (scaled by risk_weight)
        if old_condition == 0:
            risk_reward = 1.0 * self.risk_weight
        else:
            risk_reward = -10.0 * self.risk_weight
        
        # --- Cost Component ---
        action_cost = ACTION_COSTS[action] * self.cost_lambda
        cost_reward = -action_cost
        
        # --- Total Reward ---
        reward = risk_reward + cost_reward
        
        # --- State Transition ---
        if action == 0:  # DoNothing
            # Use age-adjusted transition matrix
            base_trans = self.transitions[0]
            age_adjusted_trans = self._get_age_adjusted_transition(base_trans)
            trans_probs = age_adjusted_trans[old_condition]
            new_condition = self.np_random.choice([0, 1], p=trans_probs)
        
        elif action == 1:  # Repair
            # Reset to normal with high probability (less aging effect)
            base_trans = self.transitions[1]
            age_adjusted_trans = self._get_age_adjusted_transition(base_trans)
            trans_probs = age_adjusted_trans[old_condition]
            new_condition = self.np_random.choice([0, 1], p=trans_probs)
        
        elif action == 2:  # Replace
            # Reset to normal with very high probability and reset equipment age
            base_trans = self.transitions[2]
            trans_probs = base_trans[old_condition]  # 交換時は老朽化リセット
            new_condition = self.np_random.choice([0, 1], p=trans_probs)
            self.equipment_age = 0.0  # 設備年数をリセット
        
        # Sample new temperature
        self.current_condition = new_condition
        self.current_temperature = self._sample_temperature(new_condition)
        
        # Update equipment age (except when replaced)
        if action != 2:  # Replace以外は年数を進める
            self.equipment_age += (1.0 / self.horizon)  # 1エピソード = 約1年と仮定
        
        # Step increment
        self.current_step += 1
        
        # Episode termination
        terminated = False
        truncated = self.current_step >= self.horizon
        
        # Info
        info = {
            'action': ACTION_NAMES[action],
            'old_condition': STATE_NAMES[old_condition],
            'new_condition': STATE_NAMES[new_condition],
            'temperature': self.current_temperature,
            'equipment_age': self.equipment_age,
            'aging_factor': self.aging_factor,
            'risk_reward': risk_reward,
            'cost_reward': cost_reward,
            'total_reward': reward,
            'step': self.current_step
        }
        
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """環境の描画"""
        if self.render_mode == 'human' or self.render_mode == 'ansi':
            condition_str = STATE_NAMES[self.current_condition]
            temp_str = f"{self.current_temperature:.1f}°C"
            age_str = f"{self.equipment_age:.1f}年"
            print(f"Step {self.current_step}: {condition_str}, Temp={temp_str}, Age={age_str}")


def test_environment():
    """環境の動作テスト"""
    print("="*60)
    print("🧪 Equipment CBM Environment Test")
    print("="*60)
    
    # サンプル遷移行列（データから推定したものを想定）
    transition_matrix = np.array([
        [0.96, 0.04],  # Normal → [Normal, Anomalous]
        [0.15, 0.85],  # Anomalous → [Normal, Anomalous]
    ], dtype=np.float32)
    
    env = EquipmentCBMEnvironment(
        transition_matrix=transition_matrix,
        temperature_range=(0.0, 150.0),
        normal_temp_range=(20.0, 100.0),
        horizon=20,
        seed=42,
        render_mode='human'
    )
    
    print("\n✅ Environment created")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Transition matrix:\n{transition_matrix}")
    
    # エピソード実行
    obs, info = env.reset(seed=42)
    print(f"\n🎬 Initial: condition={info['condition']}, temp={info['temperature']:.1f}°C")
    
    total_reward = 0.0
    actions_taken = []
    
    for step in range(20):
        # ランダムアクション（実際にはDQNが選択）
        if env.current_condition == 1:  # Anomalousなら修理
            action = env.action_space.sample()  # 1 or 2
            if action == 0:
                action = 1
        else:
            action = 0  # Normalなら何もしない
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        actions_taken.append(info['action'])
        
        print(f"  Action: {info['action']:10s} | "
              f"{info['old_condition']} → {info['new_condition']:10s} | "
              f"Reward: {reward:6.2f} | Temp: {info['temperature']:5.1f}°C")
        
        if terminated or truncated:
            break
    
    print(f"\n📊 Episode Summary:")
    print(f"  - Total steps: {env.current_step}")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Actions: {dict((a, actions_taken.count(a)) for a in set(actions_taken))}")


if __name__ == "__main__":
    test_environment()
