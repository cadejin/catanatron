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
from catanatron.models.enums import RESOURCES
from catanatron.state_functions import player_key

DATASET_PATH = "sft_dataset1.pkl"
MODEL_PATH = "policy_sft.pt"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ACTION2IDX = {}
IDX2ACTION = []

def encode_action(action):
    key = (action.__class__.__name__, str(action))
    if key not in ACTION2IDX:
        ACTION2IDX[key] = len(ACTION2IDX)
        IDX2ACTION.append(key)
    return ACTION2IDX[key]

def encode_state(state):
    idx = state.current_player_index
    key_prefix = f"P{idx}_"
    res = [state.player_state[f"{key_prefix}{r}_IN_HAND"] for r in RESOURCES]
    vp = state.player_state[f"{key_prefix}VICTORY_POINTS"]
    return res + [vp]

def generate_expert_data(num_games=500):
    X, Y = [], []
    player_types = [AlphaBetaPlayer, ValueFunctionPlayer]

    for _ in range(num_games):
        print(f'generating game {_}', flush=True)
        colors = random.sample(list(Color), 2)
        players_list = [random.choice(player_types)(color=c) for c in colors]
        color_to_player = {p.color: p for p in players_list}

        game = Game(players_list)

        while game.winning_color() is None:
            current_color = game.state.current_color()
            player = color_to_player[current_color]
            actions = game.state.playable_actions

            action = player.decide(game, actions)

            X.append(encode_state(game.state))
            Y.append(encode_action(action))

            game.execute(action)

    return X, Y

def load_dataset():
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "rb") as f:
            X, Y = pickle.load(f)
        print(f"Loaded dataset")
    else:
        X, Y = [], []
    return X, Y

def save_dataset(X, Y):
    with open(DATASET_PATH, "wb") as f:
        pickle.dump((X, Y), f)
    print(f"Saved dataset")

class PolicyNet(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.net(x)

def train_sft(num_new_games=500, epochs=50, max_dataset_size=10000):
    X, Y = load_dataset()

    X_new, Y_new = generate_expert_data(num_games=num_new_games)
    print(f"Generated new games")

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
    model = PolicyNet(input_dim, num_actions)

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH)
        if checkpoint["state_dict"]["net.4.weight"].shape[0] == num_actions:
            model.load_state_dict(checkpoint["state_dict"])
            print("Loaded existing model checkpoint")
        else:
            print("Existing checkpoint action size mismatch, initializing new model")

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in dl:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}: loss={total_loss/len(dl):.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "num_actions": num_actions,
        "idx2action": IDX2ACTION,
    }, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    train_sft()
