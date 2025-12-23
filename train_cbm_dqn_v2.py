"""
QR-DQN Training Script for Equipment CBM MVP (v2.0)

Full integration of base_markov-dqn-v09-quantile optimizations:
- Prioritized Experience Replay (PER) with α=0.6, β=0.4
- N-step Learning (n=3) with gamma adjustment
- Mixed Precision Training (AMP) with GradScaler
- AsyncVectorEnv parallelization (16 environments)
- Noisy Networks for parameter-space exploration
- QR-DQN (51 quantiles) for distributional RL
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import json
from gymnasium.vector import AsyncVectorEnv
from torch.amp import autocast, GradScaler
import sys

sys.path.insert(0, str(Path(__file__).parent))
from cbm_environment import EquipmentCBMEnvironment
from data_preprocessor import CBMDataPreprocessor


# ===== Noisy Networks =====

class NoisyLinear(nn.Module):
    """Noisy Linear for parameter-space exploration (Fortunato et al., 2018)"""
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x: torch.Tensor):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


# ===== QR-DQN Network =====

class CBMQRDQN(nn.Module):
    """QR-DQN with Dueling Architecture and Noisy Networks"""
    
    def __init__(self, state_dim: int, n_actions: int, n_quantiles: int = 51, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream (Noisy)
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, n_quantiles)
        )
        
        # Advantage stream (Noisy)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, n_actions * n_quantiles)
        )
    
    def forward(self, x: torch.Tensor):
        """
        Returns:
            q_values: (batch, n_actions) - mean Q-values
            quantiles: (batch, n_actions, n_quantiles) - full distributions
        """
        batch_size = x.size(0)
        features = self.feature(x)
        
        # Dueling architecture
        value = self.value_stream(features)  # (batch, n_quantiles)
        advantage = self.advantage_stream(features)  # (batch, n_actions * n_quantiles)
        
        value = value.view(batch_size, 1, self.n_quantiles)
        advantage = advantage.view(batch_size, self.n_actions, self.n_quantiles)
        
        # Combine: Q = V + (A - mean(A))
        quantiles = value + (advantage - advantage.mean(dim=1, keepdim=True))
        q_values = quantiles.mean(dim=2)  # Average over quantiles
        
        return q_values, quantiles
    
    def reset_noise(self):
        """Reset noise in all noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ===== Prioritized N-step Replay Buffer =====

class PrioritizedNStepBuffer:
    """
    Prioritized Experience Replay with N-step returns
    """
    
    def __init__(self, capacity: int, n_steps: int = 3, gamma: float = 0.95, 
                 alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # N-step buffer
        self.n_step_buffer = []
    
    def push(self, state, action, reward, next_state, done):
        """Store transition with n-step return"""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_steps:
            return
        
        # Compute n-step return
        n_step_state, n_step_action = self.n_step_buffer[0][:2]
        n_step_reward = 0.0
        for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * r
            if d:
                break
        
        n_step_next_state = self.n_step_buffer[-1][3]
        n_step_done = self.n_step_buffer[-1][4]
        
        # Store with max priority
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done))
        else:
            self.buffer[self.position] = (n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # Remove oldest from n-step buffer
        if self.n_step_buffer[0][4]:  # If done
            self.n_step_buffer.clear()
        else:
            self.n_step_buffer.pop(0)
    
    def sample(self, batch_size: int):
        """Sample batch with prioritized sampling"""
        if self.size < batch_size:
            return None
        
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Extract samples
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32)
        )
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6
    
    def __len__(self):
        return self.size


# ===== Quantile Huber Loss with PER =====

def quantile_huber_loss_per(
    policy_net, target_net, 
    states, actions, rewards, next_states, dones, 
    weights, gamma, kappa=1.0, n_steps=3
):
    """
    Compute QR-DQN loss with importance sampling weights
    
    Returns:
        loss: Weighted quantile Huber loss
        td_errors: TD errors for priority updates
    """
    batch_size = states.size(0)
    n_quantiles = policy_net.n_quantiles
    
    # Current quantiles
    _, current_quantiles = policy_net(states)
    current_quantiles = current_quantiles.gather(1, actions.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, n_quantiles)).squeeze(1)
    
    # Target quantiles (Double DQN)
    with torch.no_grad():
        next_q_values, _ = policy_net(next_states)
        next_actions = next_q_values.argmax(dim=1)
        _, next_quantiles = target_net(next_states)
        next_quantiles = next_quantiles.gather(1, next_actions.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, n_quantiles)).squeeze(1)
        
        # N-step target
        target_quantiles = rewards.unsqueeze(-1) + (gamma ** n_steps) * next_quantiles * (1 - dones.unsqueeze(-1))
    
    # Quantile Huber loss
    tau = torch.linspace(0.0, 1.0, n_quantiles + 1, device=states.device)
    tau = (tau[:-1] + tau[1:]) / 2.0
    tau = tau.view(1, 1, n_quantiles)
    
    td_errors_matrix = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
    huber_loss = torch.where(td_errors_matrix.abs() <= kappa, 
                             0.5 * td_errors_matrix ** 2,
                             kappa * (td_errors_matrix.abs() - 0.5 * kappa))
    
    quantile_loss = (tau - (td_errors_matrix < 0).float()).abs() * huber_loss
    loss_per_sample = quantile_loss.sum(dim=2).mean(dim=1)
    
    # Apply importance sampling weights
    weighted_loss = (loss_per_sample * weights).mean()
    
    # TD errors for priority updates
    td_errors = loss_per_sample.detach()
    
    return weighted_loss, td_errors


# ===== Environment Factory =====

def make_cbm_env(transition_matrix, temperature_range, normal_temp_range, 
                 horizon=100, gamma=0.95, risk_weight=1.0, cost_lambda=0.15, scenario=None,
                 equipment_age=0.0, aging_factor=0.01, max_equipment_age=50.0, seed=42):
    """Factory function for creating CBM environments"""
    def _init():
        env = EquipmentCBMEnvironment(
            transition_matrix=transition_matrix,
            temperature_range=temperature_range,
            normal_temp_range=normal_temp_range,
            horizon=horizon,
            gamma=gamma,
            risk_weight=risk_weight,
            cost_lambda=cost_lambda,
            scenario=scenario,
            equipment_age=equipment_age,
            aging_factor=aging_factor,
            max_equipment_age=max_equipment_age
        )
        env.reset(seed=seed)
        return env
    return _init


# ===== Training Loop =====

def train_cbm_dqn_parallel(
    transition_matrix: np.ndarray,
    temperature_range: tuple,
    normal_temp_range: tuple,
    n_episodes: int = 1000,
    n_envs: int = 16,
    horizon: int = 100,
    gamma: float = 0.95,
    risk_weight: float = 1.0,
    cost_lambda: float = 0.15,
    scenario: str = None,
    equipment_age: float = 0.0,
    aging_factor: float = 0.01,
    max_equipment_age: float = 50.0,
    lr: float = 1.5e-3,
    batch_size: int = 64,
    buffer_capacity: int = 10000,
    target_sync_steps: int = 500,
    n_quantiles: int = 51,
    n_steps: int = 3,
    kappa: float = 1.0,
    seed: int = 42,
    device: str = 'cuda',
    save_dir: Path = None
):
    """
    Train QR-DQN agent with full optimizations from base_markov-dqn-v09
    """
    
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*80}")
    print(f"EQUIPMENT CBM QR-DQN TRAINING (v2.0)")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Episodes: {n_episodes}, Parallel Envs: {n_envs}")
    print(f"  Device: {device}, Gamma: {gamma}, LR: {lr}")
    print(f"  Buffer: {buffer_capacity}, Batch: {batch_size}")
    print(f"  Target Sync: {target_sync_steps} steps")
    print(f"\nOptimizations:")
    print(f"  ✓ QR-DQN (Quantiles={n_quantiles})")
    print(f"  ✓ Prioritized Experience Replay (α=0.6, β=0.4)")
    print(f"  ✓ N-step Learning (n={n_steps})")
    print(f"  ✓ Mixed Precision Training (AMP)")
    print(f"  ✓ AsyncVectorEnv ({n_envs} parallel)")
    print(f"  ✓ Noisy Networks (no ε-greedy)")
    print(f"{'='*80}\n")
    
    # Create vectorized environments
    env_fns = [
        make_cbm_env(
            transition_matrix=transition_matrix,
            temperature_range=temperature_range,
            normal_temp_range=normal_temp_range,
            horizon=horizon,
            gamma=gamma,
            risk_weight=risk_weight,
            cost_lambda=cost_lambda,
            scenario=scenario,
            equipment_age=equipment_age,
            aging_factor=aging_factor,
            max_equipment_age=max_equipment_age,
            seed=seed + i
        )
        for i in range(n_envs)
    ]
    envs = AsyncVectorEnv(env_fns)
    
    # Initialize networks (updated for 3D state space)
    agent_net = CBMQRDQN(state_dim=3, n_actions=3, n_quantiles=n_quantiles).to(device)
    target_net = CBMQRDQN(state_dim=3, n_actions=3, n_quantiles=n_quantiles).to(device)
    target_net.load_state_dict(agent_net.state_dict())
    target_net.eval()
    
    optimizer = optim.AdamW(agent_net.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler('cuda') if device == 'cuda' else None
    
    buffer = PrioritizedNStepBuffer(
        buffer_capacity, n_steps=n_steps, gamma=gamma,
        alpha=0.6, beta=0.4, beta_increment=0.001
    )
    
    # Training tracking
    rewards_history = []
    losses_history = []
    total_steps = 0
    episodes_completed = 0
    
    # Reset environments
    observations, infos = envs.reset()
    states = observations.astype(np.float32)
    
    # Episode tracking per environment
    env_episode_rewards = np.zeros(n_envs)
    
    # Training loop
    print(f"\n🚀 開始学習 - エピソード数: {n_episodes}")
    print(f"   設備年数: {equipment_age:.1f}年, 劣化因子: {aging_factor:.3f}")
    
    pbar = tqdm(total=n_episodes, desc="Training")
    start_time = time.time()
    
    while episodes_completed < n_episodes:
        # Reset noise for exploration
        agent_net.reset_noise()
        
        # Select actions for all environments
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(device)
            q_values, _ = agent_net(states_t)
            actions = q_values.argmax(dim=1).cpu().numpy()
        
        # Step all environments
        next_observations, rewards, terminateds, truncateds, infos = envs.step(actions)
        next_states = next_observations.astype(np.float32)
        
        # Store transitions
        for i in range(n_envs):
            done = terminateds[i] or truncateds[i]
            buffer.push(states[i], actions[i], rewards[i], next_states[i], done)
            
            env_episode_rewards[i] += rewards[i]
            
            # Episode done
            if done and episodes_completed < n_episodes:
                rewards_history.append(env_episode_rewards[i])
                env_episode_rewards[i] = 0.0
                
                episodes_completed += 1
                pbar.update(1)
                
                # Logging
                if episodes_completed % 100 == 0:
                    avg_reward = np.mean(rewards_history[-100:])
                    avg_loss = np.mean(losses_history[-1000:]) if losses_history else 0.0
                    elapsed = time.time() - start_time
                    
                    pbar.write(f"\n📊 Episode {episodes_completed}/{n_episodes}")
                    pbar.write(f"   Avg Reward (last 100): {avg_reward:.2f}")
                    pbar.write(f"   Avg Loss (last 1000): {avg_loss:.4f}")
                    pbar.write(f"   Time: {elapsed:.1f}s ({elapsed/episodes_completed:.3f}s/ep)")
        
        states = next_states
        total_steps += n_envs
        
        # Optimization step
        if len(buffer) >= batch_size:
            sample = buffer.sample(batch_size)
            if sample is not None:
                s_b, a_b, r_b, sn_b, d_b, indices, weights = sample
                
                # Convert to tensors
                s_b_t = torch.FloatTensor(s_b).to(device)
                a_b_t = torch.LongTensor(a_b).to(device)
                r_b_t = torch.FloatTensor(r_b).to(device)
                sn_b_t = torch.FloatTensor(sn_b).to(device)
                d_b_t = torch.FloatTensor(d_b).to(device)
                w_b_t = torch.FloatTensor(weights).to(device)
                
                # Mixed precision training
                if scaler:
                    with autocast('cuda'):
                        loss, td_errors = quantile_huber_loss_per(
                            agent_net, target_net, s_b_t, a_b_t, r_b_t, sn_b_t, d_b_t,
                            w_b_t, gamma, kappa=kappa, n_steps=n_steps
                        )
                    
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(agent_net.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss, td_errors = quantile_huber_loss_per(
                        agent_net, target_net, s_b_t, a_b_t, r_b_t, sn_b_t, d_b_t,
                        w_b_t, gamma, kappa=kappa, n_steps=n_steps
                    )
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent_net.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Update priorities
                buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
                losses_history.append(loss.item())
        
        # Sync target network
        if total_steps % target_sync_steps == 0:
            target_net.load_state_dict(agent_net.state_dict())
    
    pbar.close()
    envs.close()
    
    # Training summary
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total Episodes: {episodes_completed}")
    print(f"Total Time: {elapsed_time:.2f} sec ({elapsed_time/60:.2f} min)")
    print(f"Time per Episode: {elapsed_time/episodes_completed:.3f} sec")
    print(f"Final Reward (last 100): {np.mean(rewards_history[-100:]):.2f}")
    print(f"{'='*80}\n")
    
    # Save results
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(agent_net.state_dict(), save_dir / "policy_net.pth")
        print(f"💾 Model saved: {save_dir / 'policy_net.pth'}")
        
        # Save training history
        history = {
            'episode_rewards': rewards_history,
            'episode_lengths': [horizon] * len(rewards_history),
            'losses': losses_history
        }
        with open(save_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        print(f"💾 History saved: {save_dir / 'training_history.json'}")
    
    return rewards_history, [horizon] * len(rewards_history), losses_history


# ===== Main =====

def main():
    parser = argparse.ArgumentParser(description="Equipment CBM QR-DQN Training (v2.0)")
    parser.add_argument('--equipment_id', type=int, default=43175, help='Equipment ID')
    parser.add_argument('--measurement_id', type=int, default=167473, help='Measurement ID')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--n_envs', type=int, default=16, help='Number of parallel environments')
    parser.add_argument('--horizon', type=int, default=100, help='Episode horizon')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--scenario', type=str, default='balanced', 
                       choices=['safety_first', 'cost_efficient', 'balanced'],
                       help='Maintenance scenario: safety_first (安全重視), cost_efficient (コスト重視), balanced (バランス型)')
    parser.add_argument('--risk_weight', type=float, default=None, help='Risk weight (overrides scenario)')
    parser.add_argument('--cost_lambda', type=float, default=None, help='Cost weight (overrides scenario)')
    parser.add_argument('--equipment_age', type=float, default=None, help='Initial equipment age (use real age if None)')
    parser.add_argument('--aging_factor', type=float, default=0.01, help='Annual anomaly probability increase')
    parser.add_argument('--lr', type=float, default=1.5e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--n_quantiles', type=int, default=51, help='Number of quantiles')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='outputs_cbm_v2', help='Output directory')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("="*80)
    print("🏭 Equipment CBM QR-DQN Training (v2.0 with Equipment Aging)")
    print("="*80)
    
    # Load data and estimate transition matrix
    data_dir = Path(__file__).parent / "data" / "private_benchmark"
    preprocessor = CBMDataPreprocessor(data_dir)
    preprocessor.load_data()
    
    print(f"\n📊 Processing equipment {args.equipment_id}, measurement {args.measurement_id}...")
    stats = preprocessor.process_equipment(args.equipment_id, args.measurement_id)
    
    # Prepare environment parameters
    transition_matrix = np.array(stats['transition_matrix'], dtype=np.float32)
    temp_min = stats['value_stats']['min']
    temp_max = stats['value_stats']['max']
    normal_temp_min = stats['thresholds']['Smin']
    normal_temp_max = stats['thresholds']['Smax']
    
    # Equipment aging parameters
    if args.equipment_age is None and 'age_analysis' in stats and stats['age_analysis']['age_available']:
        # 実際のデータから平均年数を取得
        equipment_age = np.mean(stats['age_analysis']['age_range'])
        print(f"📅 Using real equipment age: {equipment_age:.1f} years")
    else:
        equipment_age = args.equipment_age if args.equipment_age is not None else 0.0
        print(f"📅 Using specified equipment age: {equipment_age:.1f} years")
    
    print(f"\n✅ Environment parameters prepared")
    print(f"   Temperature range: [{temp_min:.1f}, {temp_max:.1f}]")
    print(f"   Normal range: [{normal_temp_min:.1f}, {normal_temp_max:.1f}]")
    print(f"   Equipment age: {equipment_age:.1f} years")
    print(f"   Aging factor: {args.aging_factor:.3f} per year")
    print(f"   Transition Matrix:")
    print(f"      Normal→Normal: {transition_matrix[0,0]:.4f}")
    print(f"      Normal→Anomalous: {transition_matrix[0,1]:.4f}")
    print(f"      Anomalous→Normal: {transition_matrix[1,0]:.4f}")
    print(f"      Anomalous→Anomalous: {transition_matrix[1,1]:.4f}")
    
    # Aging analysis display
    if 'age_analysis' in stats and stats['age_analysis']['age_available']:
        age_analysis = stats['age_analysis']
        print(f"   Age-Anomaly correlation: {age_analysis['overall_age_correlation']:.3f}")
        if age_analysis['degradation_trend']:
            trend = age_analysis['degradation_trend']
            print(f"   Observed degradation trend: +{trend['slope']*100:.2f}%/year")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✅ Device: {device}")
    
    # Determine scenario parameters
    from cbm_environment import MAINTENANCE_SCENARIOS
    
    if args.risk_weight is not None and args.cost_lambda is not None:
        # Custom parameters override scenario
        risk_weight = args.risk_weight
        cost_lambda = args.cost_lambda
        scenario_name = None
        print(f"\n🔧 Custom Parameters:")
        print(f"   Risk Weight: {risk_weight}, Cost Lambda: {cost_lambda}")
    else:
        # Use scenario preset
        scenario_name = args.scenario
        if scenario_name in MAINTENANCE_SCENARIOS:
            scenario_config = MAINTENANCE_SCENARIOS[scenario_name]
            risk_weight = scenario_config['risk_weight']
            cost_lambda = scenario_config['cost_lambda']
            print(f"\n📋 Maintenance Scenario: {scenario_name}")
            print(f"   {scenario_config['description']}")
            print(f"   Risk Weight: {risk_weight}, Cost Lambda: {cost_lambda}")
        else:
            # Default to balanced
            risk_weight = 1.0
            cost_lambda = 0.15
            scenario_name = 'balanced'
            print(f"\n📋 Using default 'balanced' scenario")
    
    # Train with parallel environments
    output_dir = Path(args.output_dir)
    episode_rewards, episode_lengths, losses = train_cbm_dqn_parallel(
        transition_matrix=transition_matrix,
        temperature_range=(temp_min, temp_max),
        normal_temp_range=(normal_temp_min, normal_temp_max),
        n_episodes=args.episodes,
        n_envs=args.n_envs,
        horizon=args.horizon,
        gamma=args.gamma,
        risk_weight=risk_weight,
        cost_lambda=cost_lambda,
        scenario=scenario_name,
        equipment_age=equipment_age,
        aging_factor=args.aging_factor,
        max_equipment_age=50.0,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_size,
        n_quantiles=args.n_quantiles,
        seed=args.seed,
        device=device,
        save_dir=output_dir
    )
    
    print(f"✅ Training completed! Final reward: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"📁 Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
