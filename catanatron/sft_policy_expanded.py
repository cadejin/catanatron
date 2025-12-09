import os
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.models.enums import RESOURCES, DEVELOPMENT_CARDS
from catanatron.state_functions import (
    player_key,
    get_longest_road_length,
    get_largest_army,
    get_dev_cards_in_hand,
    get_played_dev_cards,
    get_player_buildings,
)

DATASET_PATH = "sft_dataset_expanded.pkl"
MODEL_PATH = "policy_sft_expanded.pt"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ACTION2IDX = {}
IDX2ACTION = []

# Resource and dev card lists for iteration
RESOURCE_LIST = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
DEV_CARD_LIST = ["KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"]


def encode_action(action):
    key = (action.__class__.__name__, str(action))
    if key not in ACTION2IDX:
        ACTION2IDX[key] = len(ACTION2IDX)
        IDX2ACTION.append(key)
    return ACTION2IDX[key]


def encode_state(game_state):
    """
    Encode game state into a feature vector.
    
    Features include (for current player and opponent):
    - Resources in hand (5 each)
    - Victory points
    - Number of settlements, cities, roads
    - Development cards in hand (5 types)
    - Played development cards (5 types)  
    - Longest road length
    - Has longest road bonus
    - Army size (knights played)
    - Has largest army bonus
    - Total resource count
    
    Additional game state:
    - Bank resources remaining (5)
    - Turn number (normalized)
    """
    features = []
    
    # Get current player index and color
    current_idx = game_state.current_player_index
    current_color = game_state.colors[current_idx]
    
    # Get opponent index and color (assuming 2-player game)
    opponent_idx = 1 - current_idx
    opponent_color = game_state.colors[opponent_idx]
    
    # Get longest road and largest army info
    longest_road_color = None
    largest_army_color, largest_army_size = get_largest_army(game_state)
    
    # Encode features for current player
    features.extend(encode_player_features(game_state, current_color, current_idx, 
                                           longest_road_color, largest_army_color))
    
    # Encode features for opponent
    features.extend(encode_player_features(game_state, opponent_color, opponent_idx,
                                           longest_road_color, largest_army_color))
    
    # Bank resources (normalized)
    for resource in RESOURCE_LIST:
        bank_key = f"BANK_{resource}"
        if bank_key in game_state.resource_freqdeck:
            features.append(game_state.resource_freqdeck[bank_key] / 19.0)
        else:
            # Bank is stored differently - try to access it
            features.append(0.5)  # Default middle value
    
    # Turn number (normalized, assuming games rarely go past 200 turns)
    features.append(min(game_state.num_turns / 200.0, 1.0))
    
    return features


def encode_player_features(game_state, color, player_idx, longest_road_color, largest_army_color):
    """Encode features for a single player."""
    features = []
    key_prefix = f"P{player_idx}_"
    
    # Resources in hand (5 features) - normalized
    for resource in RESOURCE_LIST:
        res_key = f"{key_prefix}{resource}_IN_HAND"
        res_count = game_state.player_state.get(res_key, 0)
        features.append(res_count / 10.0)  # Normalize (rarely have >10 of one resource)
    
    # Total resources in hand
    total_resources = sum(
        game_state.player_state.get(f"{key_prefix}{r}_IN_HAND", 0) 
        for r in RESOURCE_LIST
    )
    features.append(total_resources / 20.0)
    
    # Victory points (normalized to 10, the winning condition)
    vp_key = f"{key_prefix}VICTORY_POINTS"
    vp = game_state.player_state.get(vp_key, 0)
    features.append(vp / 10.0)
    
    # Buildings count
    settlements_key = f"{key_prefix}SETTLEMENTS_AVAILABLE"
    cities_key = f"{key_prefix}CITIES_AVAILABLE"
    roads_key = f"{key_prefix}ROADS_AVAILABLE"
    
    # Settlements built = 5 - available
    settlements_available = game_state.player_state.get(settlements_key, 5)
    settlements_built = 5 - settlements_available
    features.append(settlements_built / 5.0)
    
    # Cities built = 4 - available  
    cities_available = game_state.player_state.get(cities_key, 4)
    cities_built = 4 - cities_available
    features.append(cities_built / 4.0)
    
    # Roads built = 15 - available
    roads_available = game_state.player_state.get(roads_key, 15)
    roads_built = 15 - roads_available
    features.append(roads_built / 15.0)
    
    # Development cards in hand (5 features)
    for dev_card in DEV_CARD_LIST:
        dev_key = f"{key_prefix}{dev_card}_IN_HAND"
        dev_count = game_state.player_state.get(dev_key, 0)
        features.append(dev_count / 5.0)
    
    # Played development cards - specifically knights for army
    knights_played_key = f"{key_prefix}KNIGHT_PLAYED"
    knights_played = game_state.player_state.get(knights_played_key, 0)
    features.append(knights_played / 10.0)
    
    # Longest road length
    try:
        road_length = get_longest_road_length(game_state, color)
        features.append(road_length / 15.0)
    except:
        features.append(0.0)
    
    # Has longest road bonus (2 VP)
    has_longest_road_key = f"{key_prefix}HAS_ROAD"
    has_longest_road = game_state.player_state.get(has_longest_road_key, False)
    features.append(1.0 if has_longest_road else 0.0)
    
    # Has largest army bonus (2 VP)
    has_largest_army_key = f"{key_prefix}HAS_ARMY"
    has_largest_army = game_state.player_state.get(has_largest_army_key, False)
    features.append(1.0 if has_largest_army else 0.0)
    
    # Production potential - sum of probabilities for owned tiles
    # This would require board access, so we'll use a proxy
    # (settlements + 2*cities gives rough production multiplier)
    production_proxy = (settlements_built + 2 * cities_built) / 13.0
    features.append(production_proxy)
    
    return features  # 20 features per player


def generate_expert_data(num_games=500):
    X, Y = [], []
    player_types = [AlphaBetaPlayer, WeightedRandomPlayer, VictoryPointPlayer, RandomPlayer, ValueFunctionPlayer]

    for game_num in range(num_games):
        print(f'generating game {game_num}', flush=True)
        colors = random.sample(list(Color), 2)
        players_list = [random.choice(player_types)(color=c) for c in colors]
        color_to_player = {p.color: p for p in players_list}

        game = Game(players_list)

        while game.winning_color() is None:
            current_color = game.state.current_color()
            player = color_to_player[current_color]
            actions = game.state.playable_actions

            action = player.decide(game, actions)

            try:
                state_features = encode_state(game.state)
                X.append(state_features)
                Y.append(encode_action(action))
            except Exception as e:
                # Skip this state if encoding fails
                print(f"Warning: Failed to encode state: {e}")

            game.execute(action)

    return X, Y


def load_dataset():
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "rb") as f:
            data = pickle.load(f)
            X, Y = data["X"], data["Y"]
            # Restore action mappings
            global ACTION2IDX, IDX2ACTION
            ACTION2IDX = data.get("action2idx", {})
            IDX2ACTION = data.get("idx2action", [])
        print(f"Loaded dataset with {len(X)} samples, {len(IDX2ACTION)} actions")
    else:
        X, Y = [], []
    return X, Y


def save_dataset(X, Y):
    with open(DATASET_PATH, "wb") as f:
        pickle.dump({
            "X": X, 
            "Y": Y,
            "action2idx": ACTION2IDX,
            "idx2action": IDX2ACTION
        }, f)
    print(f"Saved dataset with {len(X)} samples, {len(IDX2ACTION)} actions")


class PolicyNet(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.net(x)


def train_sft(num_new_games=500, epochs=50, max_dataset_size=50000):
    X, Y = load_dataset()

    X_new, Y_new = generate_expert_data(num_games=num_new_games)
    print(f"Generated {len(X_new)} new samples from {num_new_games} games")

    X.extend(X_new)
    Y.extend(Y_new)

    if len(X) > max_dataset_size:
        X = X[-max_dataset_size:]
        Y = Y[-max_dataset_size:]

    save_dataset(X, Y)

    X_t = torch.tensor(np.array(X), dtype=torch.float32)
    Y_t = torch.tensor(np.array(Y), dtype=torch.long)

    ds = TensorDataset(X_t, Y_t)
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    num_actions = len(ACTION2IDX)
    input_dim = X_t.shape[1]
    print(f"Training with {input_dim} input features, {num_actions} actions")
    
    model = PolicyNet(input_dim, num_actions)

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH)
        if (checkpoint["input_dim"] == input_dim and 
            checkpoint["num_actions"] == num_actions):
            model.load_state_dict(checkpoint["state_dict"])
            print("Loaded existing model checkpoint")
        else:
            print("Existing checkpoint dimension mismatch, initializing new model")

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for xb, yb in dl:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
            # Track accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        
        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs}: loss={total_loss/len(dl):.4f}, accuracy={accuracy:.2f}%")

    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "num_actions": num_actions,
        "idx2action": IDX2ACTION,
        "action2idx": ACTION2IDX,
    }, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_sft()