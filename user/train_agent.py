'''
TRAINING: AGENT

This file contains all the types of Agent classes, the Reward Function API, and the built-in train function from our multi-agent RL API for self-play training.
- All of these Agent classes are each described below.

Running this file will initiate the training function, and will:
a) Start training from scratch
b) Continue training from a specific timestep given an input `file_path`
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

import torch
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment.agent import *
from typing import Optional, Type, List, Tuple

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    Note:
    - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
    You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

class RecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 512,
                'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,

            }
            self.model = RecurrentPPO("MlpLstmPolicy",
                                      self.env,
                                      verbose=0,
                                      n_steps=30*90*20,
                                      batch_size=16,
                                      ent_coef=0.05,
                                      policy_kwargs=policy_kwargs)
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class BasedAgent(Agent):
    '''
    BasedAgent:
    - Defines a hard-coded Agent that predicts actions based on if-statements. Interesting behaviour can be achieved here.
    - The if-statement algorithm can be developed within the `predict` method below.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - Defines an Agent that performs actions entirely via real-time player input
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)

        return action

class ClockworkAgent(Agent):
    '''
    ClockworkAgent:
    - Defines an Agent that performs sequential steps of [duration, action]
    '''
    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (15, ['space']),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1  # Increment step counter
        return action

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLPExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP Base Features Extractor
    '''
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0],
            action_dim=10,
            hidden_dim=hidden_dim,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim) #NOTE: features_dim = 10 to match action space output
        )

class CustomAgent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: str = None, extractor: BaseFeaturesExtractor = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, policy_kwargs=self.extractor.get_policy_kwargs(), verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

class Skynet_Final(Agent):
    '''
    RecurrentPPOAgent:
        - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.GELU,
                'lstm_hidden_size': 256, #512
                'net_arch': [dict(pi=[128, 64, 128], vf=[128, 64, 128])],
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,

            }
            self.model = RecurrentPPO("MlpLstmPolicy",
                                      self.env,
                                      verbose=0,
                                      n_steps=30*90*20,
                                      batch_size=16,
                                      ent_coef=0.05,
                                      policy_kwargs=policy_kwargs)
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

'''
Example Reward Functions:
- Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).
'''

def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Extract the used quantities (to enable type-hinting)
    obj: GameObject = env.objects[obj_name]

    # Compute the L2 squared penalty
    return (obj.body.position.y - target_height)**2

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def fight_mode_setter(env: WarehouseBrawl) -> RewardMode:
    """
    Set FIGHT_MODE to different fighting styles depending on enemy damage.
    """
    opponent: Player = env.objects["opponent"]
    player: Player = env.objects["player"]

    if 20 <= (player.damage - opponent.damage):
        return RewardMode.ASYMMETRIC_OFFENSIVE
    elif 0 <= (player.damage - opponent.damage) < 20:
        return RewardMode.SYMMETRIC
    else:
        return RewardMode.ASYMMETRIC_DEFENSIVE

def damage_interaction_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    mode = fight_mode_setter(env)
    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward / 140


# In[ ]:


def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Determine the current fight mode to apply a conditional penalty
    mode = fight_mode_setter(env)
    current_penalty = zone_penalty
    # If the agent has a lead, penalize risky plays more heavily
    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        current_penalty *= 2

    # Apply penalty if the player is in the danger zone
    reward = -current_penalty if player.body.position.y >= zone_height else 0.0

    return reward * env.dt

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState]=BackDashState,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    mode = fight_mode_setter(env)
    if mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = 2 if isinstance(player.state, desired_state) else 0.0
    else:
        # Apply penalty if the player is in the danger zone
        reward = 1 if isinstance(player.state, desired_state) else 0.0

    return reward * env.dt

def head_to_middle_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    mode = fight_mode_setter(env)
    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        multiplier = -0.5 if player.body.position.x > 0 else 0.5
        reward = multiplier * (player.body.position.x - player.prev_x)
    else:
        # Apply penalty if the player is in the danger zone
        multiplier = -1 if player.body.position.x > 0 else 1
        reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def head_to_opponent(
    env: WarehouseBrawl,
) -> float | None:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    mode = fight_mode_setter(env)
    # Apply penalty if the player is in the danger zone
    if mode == RewardMode.SYMMETRIC:
        multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
        reward = multiplier * (player.body.position.x - player.prev_x)
        return float(reward)
    elif mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
        reward = multiplier * (player.body.position.x - player.prev_x)
        return float(reward * 2)
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
        reward = multiplier * (player.body.position.x - player.prev_x)
        return float(reward * 0.5)

def holding_more_than_3_keys(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    a = player.cur_action
    if (a > 0.5).sum() > 3:
        return env.dt
    return 0

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    else:
        return -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:

    mode = fight_mode_setter(env)

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        if agent == "player":
            if env.objects["player"].weapon == "Hammer":
                return 2.0
            elif env.objects["player"].weapon == "Spear":
                return 1.0
        return 0.0
    else:
        if agent == "player":
            if env.objects["player"].weapon == "Hammer":
                return 2.0
            elif env.objects["player"].weapon == "Spear":
                return 1.0
        return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:

    mode = fight_mode_setter(env)

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        if agent == "player":
            if env.objects["player"].weapon == "Punch":
                return -2.0
        return 0.0
    else:
        if agent == "player":
            if env.objects["player"].weapon == "Punch":
                return -1.0
        return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:

    mode = fight_mode_setter(env)

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        if agent == 'player':
            return 2.0
        else:
            return -2.0
    else:
        if agent == 'player':
            return 1.0
        else:
            return -1.0

def stock_advantage_reward(
        env: WarehouseBrawl,
        success_value: float = 1.0,
) -> float:
    """
    Computes a reward for having a stock advantage, scaled by match progression.

    This reward encourages the agent to secure and maintain a stock lead. The
    incentive grows stronger as the match nears its end, discouraging risky
    plays for an early lead while heavily incentivizing protecting that lead
    when time is running out.

    Args:
        env (WarehouseBrawl): The game environment.
        success_value (float): A base multiplier for the reward.

    Returns:
        float: The computed reward for the current timestep.
    """
    # Retrieve the player and opponent objects from the environment state
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Calculate the stock difference. A positive value means the player is ahead.
    stock_difference = player.stocks - opponent.stocks

    # If the player does not have a stock advantage, there is no reward.
    if stock_difference <= 0:
        return 0.0

    # Get the maximum number of timesteps for the match from the environment
    total_match_timesteps = env.max_timesteps

    # Avoid division by zero if the match length is not set
    if total_match_timesteps <= 0:
        return 0.0

    # Use the environment's internal step counter, assuming it's named 'timestep'.
    # This attribute would be incremented in the env.step() method.
    # If your environment uses a different name (e.g., _step, current_step),
    # replace 'timestep' with the correct attribute name.
    current_step = getattr(env, 'timestep', 0)

    # Calculate time progress as a fraction from 0.0 to 1.0.
    time_progress = min(current_step / total_match_timesteps, 1.0)

    # The reward is the product of the base value, the size of the stock lead,
    # and the time progression scaler.
    reward = success_value * stock_difference * time_progress

    # Scale the reward by the timestep duration (dt) to make it independent of the FPS.
    return reward * env.dt

def edge_guard_reward(
        env: WarehouseBrawl,
        success_value: float = 5.0,
        fail_value: float = -5.0,
        opportunity_value: float = 0.5,
        offensive_multiplier: float = 2.0,
        defensive_multiplier: float = 0.5
) -> float:
    """
    Computes a reward for edge-guarding, adapted to the current fight mode.

    An "edge-guard situation" is defined by four conditions:
    1. The opponent is horizontally off the main stage.
    2. The opponent is in an aerial state, below the stage, and recovering.
    3. The player is grounded on the stage near the correct edge.
    4. The player is facing the opponent and able to act.

    The rewards are scaled based on the fight mode:
    - OFFENSIVE: Increased rewards to encourage finishing the opponent.
    - DEFENSIVE: Decreased rewards to promote safer, less committal play.
    - SYMMETRIC: Uses the base reward values.

    Args:
        env (WarehouseBrawl): The game environment.
        success_value (float): Base reward for hitting the opponent.
        fail_value (float): Base penalty for being hit by the opponent.
        opportunity_value (float): Base reward for maintaining position.
        offensive_multiplier (float): Multiplier for rewards in OFFENSIVE mode.
        defensive_multiplier (float): Multiplier for rewards in DEFENSIVE mode.

    Returns:
        float: The computed, mode-adjusted reward for the current timestep.
    """
    # Get player and opponent objects
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # --- 1. Define Stage Geometry and Character States ---
    stage_edge_x = env.stage_width_tiles / 2.0
    main_stage_floor_y = 2.5
    guard_zone_width = 5.0

    # --- 2. Evaluate Edge-Guarding Conditions ---
    is_opponent_off_stage = abs(opponent.body.position.x) > stage_edge_x
    is_opponent_recovering = (
            isinstance(opponent.state, InAirState) and
            opponent.body.position.y > main_stage_floor_y and
            opponent.state.vulnerable()
    )
    is_player_positioned = (
            isinstance(player.state, GroundState) and
            (player.body.position.x * opponent.body.position.x) > 0 and
            (stage_edge_x - abs(player.body.position.x)) < guard_zone_width
    )
    player_facing_int = int(player.facing)
    opponent_is_right = opponent.body.position.x > player.body.position.x
    is_player_facing_opponent = (player_facing_int == 1 and opponent_is_right) or \
                                (player_facing_int == -1 and not opponent_is_right)
    can_player_act = player.state.can_control()

    # --- 3. Calculate Reward Based on Conditions and Mode ---
    reward = 0.0

    # Check if all conditions for an "edge-guard situation" are met
    if is_opponent_off_stage and is_opponent_recovering and is_player_positioned and is_player_facing_opponent and can_player_act:

        # Determine current fight mode
        mode = fight_mode_setter(env)

        # Adjust reward values based on the mode
        current_success = success_value
        current_fail = fail_value
        current_opportunity = opportunity_value

        if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
            current_success *= offensive_multiplier
            current_fail *= offensive_multiplier
            current_opportunity *= offensive_multiplier
        elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
            current_success *= defensive_multiplier
            current_fail *= defensive_multiplier
            current_opportunity *= defensive_multiplier

        # Success: Player dealt damage
        if opponent.damage_taken_this_frame > 0:
            reward = current_success
        # Failure: Player took damage
        elif player.damage_taken_this_frame > 0:
            reward = current_fail
        # Opportunity: Player is in position
        else:
            reward = current_opportunity

    # Scale the final reward by the timestep
    return reward * env.dt

def first_hit(
        env: WarehouseBrawl,
        agent: str = "player",
        success_value: float = 20.0,
        fail_value: float = 10.0,
) -> float:

    """
    Computes the reward based on who lands the first hit

    Args:
        env (WarehouseBrawl): The game environment
        agent (str): The agent that hit first ("player" or "opponent")
        success_value (float): Reward value for the player hitting first
        fail_value (float): Penalty for the opponent hitting first

    Returns:
        float: The computed reward.
    """

    reward = success_value if agent == "player" else -fail_value
    return reward

'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager():
    reward_functions = {
        'target_height_reward': RewTerm(func=base_height_l2, weight=0.05, params={'target_height': -4, 'obj_name': 'player'}),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.5),
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=1.0),
        'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.01),
        'head_to_opponent': RewTerm(func=head_to_opponent, weight=0.05),
        'penalize_attack_reward': RewTerm(func=in_state_reward, weight=-0.04, params={'desired_state': AttackState}),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.01),
        'taunt_reward': RewTerm(func=in_state_reward, weight=0.2, params={'desired_state': TauntState}),
        'stock_advantage_reward': RewTerm(func=stock_advantage_reward, weight=1.0),
        'edge_guard_reward': RewTerm(func=edge_guard_reward, weight=0.5),
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=50)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=8)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=5)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=10)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=15)),
        'first_hit_reward': ('first_hit_signal', RewTerm(func=first_hit, weight=10, params={'agent': 'player'})),
    }
    return RewardManager(reward_functions, signal_subscriptions)

# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------
'''
The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
'''
if __name__ == '__main__':
    # Create agent
    my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor)

    # Start here if you want to train from scratch. e.g:
    #my_agent = RecurrentPPOAgent()

    # Start here if you want to train from a specific timestep. e.g:
    #my_agent = RecurrentPPOAgent(file_path='checkpoints/experiment_3/rl_model_120006_steps.zip')

    # Reward manager
    reward_manager = gen_reward_manager()
    # Self-play settings
    selfplay_handler = SelfPlayRandom(
        partial(type(my_agent)), # Agent class and its keyword arguments
                                 # type(my_agent) = Agent class
    )

    # Set save settings here:
    save_handler = SaveHandler(
        agent=my_agent, # Agent to save
        save_freq=100_000, # Save frequency
        max_saved=40, # Maximum number of saved models
        save_path='checkpoints', # Save path
        run_name='experiment_1_DG',
        mode=SaveHandlerMode.FORCE # Save mode, FORCE or RESUME
    )

    # Set opponent settings here:
    opponent_specification = {
                    'self_play': (8, selfplay_handler),
                    'constant_agent': (0.5, partial(ConstantAgent)),
                    'based_agent': (1.5, partial(BasedAgent)),
                }
    opponent_cfg = OpponentsCfg(opponents=opponent_specification)

    train(my_agent,
        reward_manager,
        save_handler,
        opponent_cfg,
        CameraResolution.LOW,
        train_timesteps=1_000_000_000,
        train_logging=TrainLogging.PLOT
    )
