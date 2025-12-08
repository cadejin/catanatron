import torch
import random
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.game import Game

model_path = "policy_sft.pt"
checkpoint = torch.load(model_path)

class PolicyNet(torch.nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.net(x)

model = PolicyNet(checkpoint["input_dim"], checkpoint["num_actions"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()
idx2action = checkpoint["idx2action"]

RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
def encode_state(state):
    idx = state.current_player_index
    key_prefix = f"P{idx}_"
    res = [state.player_state[f"{key_prefix}{r}_IN_HAND"] for r in RESOURCES]
    vp = state.player_state[f"{key_prefix}VICTORY_POINTS"]
    return res + [vp]

class PolicyPlayer:
    def __init__(self, color, model, idx2action):
        self.color = color
        self.model = model
        self.idx2action = idx2action

    def decide(self, game, actions):
        if game.state.current_color() != self.color:
            raise RuntimeError("PolicyPlayer called out of turn")

        state_vec = encode_state(game.state)
        x = torch.tensor([state_vec], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)
        idx = torch.argmax(logits, dim=1).item()

        action_classname, action_str = self.idx2action[idx]
        for a in actions:
            if (a.__class__.__name__, str(a)) == (action_classname, action_str):
                return a

        return random.choice(actions)

def play_match(playerA_cls, playerB_cls, num_games=10):
    wins = {Color.RED: 0, Color.BLUE: 0}
    for game_idx in range(num_games):
        if game_idx % 2 == 0:
            red_player = playerA_cls(Color.RED)
            blue_player = playerB_cls(Color.BLUE)
        else:
            red_player = playerB_cls(Color.RED)
            blue_player = playerA_cls(Color.BLUE)

        game = Game([red_player, blue_player])

        while game.winning_color() is None:
            current_color = game.state.current_color()
            current_player = red_player if current_color == Color.RED else blue_player
            action = current_player.decide(game, game.state.playable_actions)
            game.execute(action)

        winner = game.winning_color()
        wins[winner] += 1

    return wins[Color.RED], wins[Color.BLUE]

if __name__ == "__main__":
    print("Loaded SFT model")
    print("Evaluating SFT Policy")
    
    PolicyBot = lambda color: PolicyPlayer(color, model, idx2action)
    RandomBot = lambda color: RandomPlayer(color)
    WeightedBot = lambda color: WeightedRandomPlayer(color)
    VPBot = lambda color: VictoryPointPlayer(color)
    ABBot = lambda color: AlphaBetaPlayer(color)

    red_wins, blue_wins = play_match(PolicyBot, RandomBot, num_games=200)
    print(f"PolicyBot vs RandomPlayer: wins={red_wins}, losses={blue_wins}")

    red_wins, blue_wins = play_match(PolicyBot, WeightedBot, num_games=200)
    print(f"PolicyBot vs WeightedRandomPlayer: wins={red_wins}, losses={blue_wins}")

    red_wins, blue_wins = play_match(PolicyBot, VPBot, num_games=200)
    print(f"PolicyBot vs VictoryPointPlayer: wins={red_wins}, losses={blue_wins}")

    red_wins, blue_wins = play_match(PolicyBot, ABBot, num_games=200)
    print(f"PolicyBot vs AlphaBetaPlayer: wins={red_wins}, losses={blue_wins}")
