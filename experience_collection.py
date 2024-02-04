import torch 
import gymnasium as gym 
from dataclasses import dataclass 

@dataclass
class Episode: 
    observations: torch.ByteTensor
    rewards: torch.FloatTensor
    ends: torch.LongTensor 
    actions: torch.FloatTensor
    

env = gym.make("CarRacing-v2")
number_of_episodes: int = 100 
max_number_of_steps: int = 200

observations = []
rewards = []
ends = [] 
actions = [] 

observation, info = env.reset()
for i in range(number_of_episodes):
    
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    observations.append(torch.ByteTensor(observation))
    rewards.append(torch.FloatTensor([reward]))
    ends.append(torch.LongTensor([int(terminated)]))
    actions.append(torch.FloatTensor(action))
    break 
    if terminated or truncated:
        observation, info = env.reset()
    
env.close()