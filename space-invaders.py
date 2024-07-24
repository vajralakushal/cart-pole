#import statements
import numpy as np
import gymnasium as gym

# we'll put our policy in this function
def policy(observation) -> int:

    #dimensions of the observation
    screen_height, screen_width = observation.shape

    #assuming the player is in the bottom quadrant of the screen
    player_height_bounds = screen_height // 4
    
    

    return 0



env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"action = {action}\nobservation = {observation}\nreward = {reward}\nterminated = {terminated}\ntruncated = {truncated}\ninfo = {info}")


    if terminated or truncated:
        observation, info = env.reset()

env.close()