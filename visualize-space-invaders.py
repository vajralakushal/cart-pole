import matplotlib.pyplot as plt
import gymnasium as gym

def visualize_observation(observation):
    plt.figure(figsize=(10, 8))
    plt.imshow(observation, cmap='gray')
    plt.title('Space Invaders Observation')
    plt.axis('off')
    plt.show()


env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Visualize every 100 steps (adjust as needed)
    if _ % 100 == 0:
        visualize_observation(observation)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
