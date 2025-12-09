import torch
import random
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.game import Game
from catanatron.state_functions import get_longest_road_length, get_largest_army

try:
    from catanatron.cli import register_cli_player
    HAS_CLI = True
except ImportError:
    HAS_CLI = False

# Resource and dev card lists for iteration
RESOURCE_LIST = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
DEV_CARD_LIST = ["KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"]

model_path = "policy_sft_expanded.pt"
checkpoint = torch.load(model_path)


class PolicyNet(torch.nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
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


def encode_player_features(game_state, color, player_idx, longest_road_color, largest_army_color):
    """Encode features for a single player."""
    features = []
    key_prefix = f"P{player_idx}_"
    
    # Resources in hand (5 features) - normalized
    for resource in RESOURCE_LIST:
        res_key = f"{key_prefix}{resource}_IN_HAND"
        res_count = game_state.player_state.get(res_key, 0)
        features.append(res_count / 10.0)
    
    # Total resources in hand
    total_resources = sum(
        game_state.player_state.get(f"{key_prefix}{r}_IN_HAND", 0) 
        for r in RESOURCE_LIST
    )
    features.append(total_resources / 20.0)
    
    # Victory points
    vp_key = f"{key_prefix}VICTORY_POINTS"
    vp = game_state.player_state.get(vp_key, 0)
    features.append(vp / 10.0)
    
    # Buildings count
    settlements_key = f"{key_prefix}SETTLEMENTS_AVAILABLE"
    cities_key = f"{key_prefix}CITIES_AVAILABLE"
    roads_key = f"{key_prefix}ROADS_AVAILABLE"
    
    settlements_available = game_state.player_state.get(settlements_key, 5)
    settlements_built = 5 - settlements_available
    features.append(settlements_built / 5.0)
    
    cities_available = game_state.player_state.get(cities_key, 4)
    cities_built = 4 - cities_available
    features.append(cities_built / 4.0)
    
    roads_available = game_state.player_state.get(roads_key, 15)
    roads_built = 15 - roads_available
    features.append(roads_built / 15.0)
    
    # Development cards in hand (5 features)
    for dev_card in DEV_CARD_LIST:
        dev_key = f"{key_prefix}{dev_card}_IN_HAND"
        dev_count = game_state.player_state.get(dev_key, 0)
        features.append(dev_count / 5.0)
    
    # Knights played
    knights_played_key = f"{key_prefix}KNIGHT_PLAYED"
    knights_played = game_state.player_state.get(knights_played_key, 0)
    features.append(knights_played / 10.0)
    
    # Longest road length
    try:
        road_length = get_longest_road_length(game_state, color)
        features.append(road_length / 15.0)
    except:
        features.append(0.0)
    
    # Has longest road bonus
    has_longest_road_key = f"{key_prefix}HAS_ROAD"
    has_longest_road = game_state.player_state.get(has_longest_road_key, False)
    features.append(1.0 if has_longest_road else 0.0)
    
    # Has largest army bonus
    has_largest_army_key = f"{key_prefix}HAS_ARMY"
    has_largest_army = game_state.player_state.get(has_largest_army_key, False)
    features.append(1.0 if has_largest_army else 0.0)
    
    # Production proxy
    production_proxy = (settlements_built + 2 * cities_built) / 13.0
    features.append(production_proxy)
    
    return features


def encode_state(game_state):
    """Encode game state into a feature vector."""
    features = []
    
    current_idx = game_state.current_player_index
    current_color = game_state.colors[current_idx]
    opponent_idx = 1 - current_idx
    opponent_color = game_state.colors[opponent_idx]
    
    longest_road_color = None
    largest_army_color, largest_army_size = get_largest_army(game_state)
    
    # Current player features
    features.extend(encode_player_features(game_state, current_color, current_idx,
                                           longest_road_color, largest_army_color))
    
    # Opponent features
    features.extend(encode_player_features(game_state, opponent_color, opponent_idx,
                                           longest_road_color, largest_army_color))
    
    # Bank resources
    for resource in RESOURCE_LIST:
        bank_key = f"BANK_{resource}"
        if bank_key in game_state.resource_freqdeck:
            features.append(game_state.resource_freqdeck[bank_key] / 19.0)
        else:
            features.append(0.5)
    
    # Turn number
    features.append(min(game_state.num_turns / 200.0, 1.0))
    
    return features


class PolicyPlayer:
    def __init__(self, color, model, idx2action):
        self.color = color
        self.model = model
        self.idx2action = idx2action
        self.fallback_count = 0
        self.total_decisions = 0

    def decide(self, game, actions):
        if game.state.current_color() != self.color:
            raise RuntimeError("PolicyPlayer called out of turn")

        self.total_decisions += 1
        
        state_vec = encode_state(game.state)
        x = torch.tensor([state_vec], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)
        
        # Get sorted action indices by probability
        probs = torch.softmax(logits, dim=1)
        sorted_indices = torch.argsort(probs, dim=1, descending=True)[0]
        
        # Try to find best valid action
        for idx in sorted_indices.tolist():
            if idx < len(self.idx2action):
                action_classname, action_str = self.idx2action[idx]
                for a in actions:
                    if (a.__class__.__name__, str(a)) == (action_classname, action_str):
                        return a
        
        # Fallback to random if no match found
        self.fallback_count += 1
        return random.choice(actions)

    def reset_state(self):
        """Hook for resetting state between games"""
        self.fallback_count = 0
        self.total_decisions = 0
    
    def get_fallback_rate(self):
        if self.total_decisions == 0:
            return 0.0
        return self.fallback_count / self.total_decisions * 100


def play_match(playerA_cls, playerB_cls, num_games=10, verbose=False):
    """
    Play matches between two player types.
    Returns (playerA_wins, playerB_wins) where playerA is always evaluated.
    Alternates colors to ensure fairness.
    """
    playerA_wins = 0
    playerB_wins = 0
    total_fallbacks = 0
    total_decisions = 0
    
    for game_idx in range(num_games):
        # Alternate which player gets which color
        if game_idx % 2 == 0:
            red_player = playerA_cls(Color.RED)
            blue_player = playerB_cls(Color.BLUE)
            playerA_color = Color.RED
        else:
            red_player = playerB_cls(Color.RED)
            blue_player = playerA_cls(Color.BLUE)
            playerA_color = Color.BLUE

        game = Game([red_player, blue_player])

        while game.winning_color() is None:
            current_color = game.state.current_color()
            current_player = red_player if current_color == Color.RED else blue_player
            action = current_player.decide(game, game.state.playable_actions)
            game.execute(action)

        winner = game.winning_color()
        if winner == playerA_color:
            playerA_wins += 1
        else:
            playerB_wins += 1
        
        # Track fallback rates for policy players
        for player in [red_player, blue_player]:
            if hasattr(player, 'fallback_count'):
                total_fallbacks += player.fallback_count
                total_decisions += player.total_decisions
        
        if verbose and (game_idx + 1) % 10 == 0:
            print(f"  Game {game_idx + 1}: PlayerA={playerA_wins}, PlayerB={playerB_wins}")

    fallback_rate = (total_fallbacks / total_decisions * 100) if total_decisions > 0 else 0
    return playerA_wins, playerB_wins, fallback_rate


def SFTPlayer(color):
    return PolicyPlayer(color, model, idx2action)


if HAS_CLI:
    register_cli_player("SFT", SFTPlayer)


if __name__ == "__main__":
    print(f"Loaded SFT model with {checkpoint['input_dim']} features, {checkpoint['num_actions']} actions")
    print("Evaluating SFT Policy (expanded features)\n")
    
    PolicyBot = lambda color: PolicyPlayer(color, model, idx2action)
    RandomBot = lambda color: RandomPlayer(color)
    WeightedBot = lambda color: WeightedRandomPlayer(color)
    VPBot = lambda color: VictoryPointPlayer(color)
    ABBot = lambda color: AlphaBetaPlayer(color)

    num_games = 200
    
    print(f"Running {num_games} games per matchup...\n")
    
    wins, losses, fallback = play_match(PolicyBot, RandomBot, num_games=num_games)
    print(f"PolicyBot vs RandomPlayer: wins={wins}, losses={losses}, win_rate={wins/num_games*100:.1f}%, fallback_rate={fallback:.1f}%")

    wins, losses, fallback = play_match(PolicyBot, WeightedBot, num_games=num_games)
    print(f"PolicyBot vs WeightedRandomPlayer: wins={wins}, losses={losses}, win_rate={wins/num_games*100:.1f}%, fallback_rate={fallback:.1f}%")

    wins, losses, fallback = play_match(PolicyBot, VPBot, num_games=num_games)
    print(f"PolicyBot vs VictoryPointPlayer: wins={wins}, losses={losses}, win_rate={wins/num_games*100:.1f}%, fallback_rate={fallback:.1f}%")

    wins, losses, fallback = play_match(PolicyBot, ABBot, num_games=num_games)
    print(f"PolicyBot vs AlphaBetaPlayer: wins={wins}, losses={losses}, win_rate={wins/num_games*100:.1f}%, fallback_rate={fallback:.1f}%")