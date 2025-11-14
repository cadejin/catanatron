"""
Policy Gradient RL Bot for Catanatron using REINFORCE algorithm
"""
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from catanatron import Player
from catanatron.cli import register_cli_player
from catanatron.features import create_sample_vector, get_feature_ordering
from catanatron.gym.envs.catanatron_env import ACTIONS_ARRAY
from catanatron.models.enums import Action, ActionType

FEATURES = get_feature_ordering(num_players=2)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_models", "pg_bot_v1.keras")

try:
    POLICY_MODEL = keras.models.load_model(MODEL_PATH)
    print(f"✓ Loaded policy model from {MODEL_PATH}")
except:
    POLICY_MODEL = None
    print(f"✗ No model found at {MODEL_PATH} - train first")


def normalize_action(action):
    """Normalize action for consistent encoding."""
    if action.action_type == ActionType.ROLL:
        return Action(action.color, action.action_type, None)
    elif action.action_type == ActionType.MOVE_ROBBER:
        return Action(action.color, action.action_type, action.value[0])
    elif action.action_type == ActionType.BUILD_ROAD:
        return Action(action.color, action.action_type, tuple(sorted(action.value)))
    elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return Action(action.color, action.action_type, None)
    elif action.action_type == ActionType.DISCARD:
        return Action(action.color, action.action_type, None)
    return action


def action_to_index(action):
    """Convert action to index in ACTIONS_ARRAY."""
    normalized = normalize_action(action)
    action_tuple = (normalized.action_type, normalized.value)
    try:
        return ACTIONS_ARRAY.index(action_tuple)
    except ValueError:
        return 0


class PolicyGradientBot(Player):
    """RL Bot using Policy Gradient (REINFORCE)."""
    
    def __init__(self, color):
        super().__init__(color)
        if POLICY_MODEL is None:
            raise ValueError("Model not loaded! Train it first.")
    
    def decide(self, game, playable_actions):
        """Sample action from policy distribution."""
        if len(playable_actions) == 1:
            return playable_actions[0]
        
        # Get state
        state = create_sample_vector(game, self.color, FEATURES)
        
        # Get valid action indices
        valid_indices = [action_to_index(a) for a in playable_actions]
        
        # Predict action logits
        logits = POLICY_MODEL.predict(state.reshape(1, -1), verbose=0)[0]
        
        # Mask invalid actions
        masked_logits = np.full(len(ACTIONS_ARRAY), -1e9)
        for idx in valid_indices:
            if idx < len(masked_logits):
                masked_logits[idx] = logits[idx]
        
        # Sample from softmax distribution
        probs = tf.nn.softmax(masked_logits).numpy()
        chosen_idx = np.random.choice(len(probs), p=probs)
        
        # Find corresponding action
        for i, action in enumerate(playable_actions):
            if action_to_index(action) == chosen_idx:
                return action
        
        # Fallback: return first action
        return playable_actions[0]


register_cli_player("PG", PolicyGradientBot)


if __name__ == "__main__":
    print("✓ PolicyGradientBot loaded")
    print("✓ Use 'PG' in catanatron-play")