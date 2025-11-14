"""
Train Policy Gradient Bot using REINFORCE algorithm
"""
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from catanatron import Game, RandomPlayer, Color, Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.features import create_sample_vector, get_feature_ordering
from catanatron.gym.envs.catanatron_env import ACTIONS_ARRAY
from catanatron.models.enums import ActionType

# Config
NUM_PLAYERS = 2
FEATURES = get_feature_ordering(num_players=NUM_PLAYERS)
STATE_DIM = len(FEATURES)
ACTION_DIM = len(ACTIONS_ARRAY)

# Training params
NUM_EPISODES = 2000  # Number of games to play
LEARNING_RATE = 0.0003
GAMMA = 0.99  # Discount factor
MODEL_OUTPUT = "trained_models/pg_bot_v1.keras"

print(f"State dim: {STATE_DIM}")
print(f"Action dim: {ACTION_DIM}")


def build_policy_network():
    """Build policy network that outputs action logits."""
    model = keras.Sequential([
        layers.Input(shape=(STATE_DIM,)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dense(ACTION_DIM)  # Raw logits, no activation
    ])
    return model


def normalize_action(action):
    """Normalize action."""
    if action.action_type == ActionType.ROLL:
        return (action.action_type, None)
    elif action.action_type == ActionType.MOVE_ROBBER:
        return (action.action_type, action.value[0])
    elif action.action_type == ActionType.BUILD_ROAD:
        return (action.action_type, tuple(sorted(action.value)))
    elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return (action.action_type, None)
    elif action.action_type == ActionType.DISCARD:
        return (action.action_type, None)
    return (action.action_type, action.value)


def action_to_index(action):
    """Convert action to index."""
    action_tuple = normalize_action(action)
    try:
        return ACTIONS_ARRAY.index(action_tuple)
    except ValueError:
        return 0


class PolicyGradientTrainer(keras.Model):
    """Trainable policy gradient model."""
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.policy_net = build_policy_network()
        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    def call(self, states):
        return self.policy_net(states)
    
    def get_action(self, state, valid_actions):
        """Sample action from policy."""
        state_array = np.array(state).reshape(1, -1)
        logits = self.policy_net(state_array)[0]
        
        # Mask invalid actions
        valid_indices = [action_to_index(a) for a in valid_actions]
        masked_logits = np.full(ACTION_DIM, -1e9)
        for idx in valid_indices:
            if idx < ACTION_DIM:
                masked_logits[idx] = logits[idx].numpy()
        
        # Sample
        probs = tf.nn.softmax(masked_logits).numpy()
        action_idx = np.random.choice(len(probs), p=probs)
        
        # Find action
        for action in valid_actions:
            if action_to_index(action) == action_idx:
                return action, action_idx, masked_logits[action_idx]
        
        return valid_actions[0], action_to_index(valid_actions[0]), masked_logits[action_to_index(valid_actions[0])]
    
    def update(self, states, actions, rewards):
      """Update policy using REINFORCE with entropy bonus."""
      states = np.array(states)
      actions = np.array(actions)
      rewards = np.array(rewards)
      
      # Compute discounted returns
      returns = []
      G = 0
      for r in reversed(rewards):
          G = r + GAMMA * G
          returns.insert(0, G)
      returns = np.array(returns)
      
      # Normalize returns
      returns = (returns - returns.mean()) / (returns.std() + 1e-8)
      
      with tf.GradientTape() as tape:
          logits = self.policy_net(states)
          probs = tf.nn.softmax(logits)
          log_probs = tf.nn.log_softmax(logits)
          
          action_log_probs = tf.gather_nd(
              log_probs,
              tf.stack([tf.range(len(actions)), actions], axis=1)
          )
          
          # Entropy bonus for exploration
          entropy = -tf.reduce_sum(probs * log_probs, axis=1)
          entropy_bonus = 0.01 * tf.reduce_mean(entropy)  # Small bonus for being random
          
          # Policy gradient loss - subtract entropy to maximize it
          loss = -tf.reduce_mean(action_log_probs * returns) - entropy_bonus
      
      # Update
      grads = tape.gradient(loss, self.policy_net.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
      
      return loss.numpy()


class TrainablePlayer(Player):
    """Player that uses trainable policy."""
    
    def __init__(self, color, trainer):
        super().__init__(color)
        self.trainer = trainer
        self.episode_states = []
        self.episode_actions = []
    
    def decide(self, game, playable_actions):
      state = create_sample_vector(game, self.color, FEATURES)
      state_array = np.array(state)
      action, action_idx, _ = self.trainer.get_action(state_array, playable_actions)
      
      self.episode_states.append(state_array)
      self.episode_actions.append(action_idx)
      
      return action


def play_episode(trainer):
    """Play one episode and collect trajectories."""
    trainable = TrainablePlayer(Color.RED, trainer)
    opponent = RandomPlayer(Color.BLUE)
    
    game = Game([trainable, opponent])
    winner = game.play()
    
    # Gentler reward shaping
    final_state = game.state
    my_vp = final_state.player_state[f"P0_ACTUAL_VICTORY_POINTS"]
    opp_vp = final_state.player_state[f"P1_ACTUAL_VICTORY_POINTS"]
    
    vp_diff = (my_vp - opp_vp) / 10.0  # Scale down VP difference
    
    if winner == Color.RED:
        reward = 1.0 + vp_diff  # Win + small VP bonus
    else:
        reward = -1.0 + vp_diff  # Loss + credit for VP progress
    
    # Clip rewards to prevent extremes
    reward = np.clip(reward, -2.0, 2.0)
    
    rewards = [0.0] * (len(trainable.episode_states) - 1) + [reward]
    
    return trainable.episode_states, trainable.episode_actions, rewards, reward


def main():
    print("=" * 60)
    print("Policy Gradient Training (REINFORCE)")
    print("=" * 60)
    
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    
    # Initialize trainer
    trainer = PolicyGradientTrainer(STATE_DIM, ACTION_DIM)
    
    # Training loop
    win_count = 0
    losses = []
    
    print("\nTraining...")
    for episode in tqdm(range(NUM_EPISODES)):
        # Play episode
        states, actions, rewards, final_reward = play_episode(trainer)
        
        if len(states) == 0:
            continue
        
        # Update policy
        loss = trainer.update(states, actions, rewards)
        losses.append(loss)
        
        # Track wins
        if final_reward > 0:
            win_count += 1
        
        # Log progress
        if (episode + 1) % 100 == 0:
            win_rate = win_count / 100
            avg_loss = np.mean(losses[-100:])
            print(f"\nEpisode {episode + 1}")
            print(f"  Win rate (last 100): {win_rate:.2%}")
            print(f"  Avg loss: {avg_loss:.4f}")
            win_count = 0
    
    # Save model
    trainer.policy_net.save(MODEL_OUTPUT)
    print(f"\n✓ Model saved to {MODEL_OUTPUT}")
    print("\nTest with: catanatron-play --players=PG,R --num=10")


if __name__ == "__main__":
    main()