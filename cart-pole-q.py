import numpy as np
import gymnasium as gym
import random

#https://gymnasium.farama.org/environments/classic_control/cart_pole/
env = gym.make("CartPole-v1")

#https://gymnasium.farama.org/api/spaces/fundamental/
action_space_size = env.action_space.n 
#state_space_size = env.observation_space.shape[0] # this is output as a tuple

#print(action_space_size)
#print(state_space_size)

# now we make the q table. to my understanding it's just a table of q values for every state/action pair. Q values are kinda like expected return for taking xyz action?
#however, for cartpole, since it's a continuous observation space, i'll have to discretize them. Thanks ChatGPT

num_bins = 20

# Discretize the state space
state_bins = [np.linspace(-4.8, 4.8, num_bins),    # Cart Position
              np.linspace(-3, 3, num_bins),        # Cart Velocity (heuristic limits)
              np.linspace(-0.418, 0.418, num_bins),# Pole Angle
              np.linspace(-3, 3, num_bins)]        # Pole Angular Velocity (heuristic limits)

def discretize_state(state):
    discretized = []
    for i in range(len(state)):
        discretized.append(np.digitize(state[i], state_bins[i]) - 1)
    return tuple(discretized)

#actually time for the q_table
q_table = np.zeros((num_bins, num_bins, num_bins, num_bins
                    , action_space_size))
#print(q_table)

# here, i'll define all of the params we'll use for Q learning
num_episodes = int(input("how many episodes would you like? Please answer in increments of 5000: ")) # how many games of cart-pole we'll play
max_steps = 450 # how many time steps per episode before we stop it. it's a bit weird since there is no "end" for cart-pole.
# look at documentation for a better explanation of termination and truncation. Leaving it at 450 since truncation is 500
learning_rate = float(input("learning rate? pick a  number between 0 and 1, but generally closer to 0: ")) # how much will we adjust for a new q_value when updating the q_table?
discount_rate = float(input("discount rate? pick a  number between 0 and 1, but generally closer to 1: ")) # if you've heard of time value of money, it's how much we discount future rewards in favor of current rewards
epsilon = 1 # exploration rate, how much the model will explore randomly to try to find something compared to trying to exploit the current understanding of the environment
max_epsilon = 1
min_epsilon = float(input("minimum epsilon? pick a  number between 0 and 1, but generally closer to 0: ")) # atp, it's pretty much just exploiting
epsilon_decay = float(input("epsilon decay? pick a  number between 0 and 1, but generally closer to 0: ")) # how much will the epsilon decrease. we want it to eventually go lower so that it starts to exploit its understanding after some time

print("training...")

rewards_across_episodes = [] # helpful to see over time
#ep_num = 0
for episode in range(num_episodes):
    if episode % 1000 == 0:
        print(f"episode num is {episode}")
    state = env.reset() # after every episode, reset the environment to begin anew
    state = state[0] #extracting from the tuple this returns
    #print(state)
    state = discretize_state(state)

    terminated = False # are we done with this episode?
    curr_episode_reward = 0

    for step in range(max_steps): # iterating through timesteps per episode
        #do we explore or exploit? let's randomly pick
        exp_threshold = random.uniform(0, 1) #explore/exploitation threshold

        if exp_threshold > epsilon: #exploiting criteria
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()
        
        new_state, reward, terminated, truncated, info = env.step(action)

        #updating q table
        #new_state = new_state[0]
        
        new_state = discretize_state(new_state) #discretize it for bins for easier access with a continuous distribution
        #print(f"new state is: {new_state}")
        q_table[state][action] += learning_rate * (reward + discount_rate*np.argmax(q_table[new_state]) - q_table[state][action])


        state = new_state
        curr_episode_reward += reward

        if terminated is True:
            break

        
    
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode) #adjusting our epsilon greedy strategy

    rewards_across_episodes.append(curr_episode_reward) #let's keep track of it
per = 1000
#print(len(rewards_across_episodes))
rewards_per = np.split(np.array(rewards_across_episodes), num_episodes//per)
# count = per
# print(f"Average reward per {per} eps")
# for rew in rewards_per:
#     print(count, ": ", str(sum(rew/per)))
#     count += per
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, len(rewards_per), len(rewards_across_episodes)//per)

y = np.zeros(len(rewards_per))
for i in range(len(rewards_per)):
    #print(f"average is {np.average(rewards_per[i])}")
    y[i] = np.average(rewards_per[i])

#print(f"x is {len(x)} y is {len(y)}")
#print(f"y is {y}")

plt.scatter(x, y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

params = "num_eps " + str(num_episodes) + " learning_rate " + str(learning_rate) + " discount_rate " + str(discount_rate) + " min_epsilon " + str(min_epsilon) + " decay " + str(epsilon_decay)
plt.title(params)
plt.plot(x,p(x),"r--")

plt.show()

#want to save your q table?
# q_table_choice = input("would you like to save your q table: Y/N: ")
# if q_table_choice == 'Y' or q_table_choice == 'y':
#     np.savetxt("q_table.txt", q_table)