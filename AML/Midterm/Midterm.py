"""
Summarize the problem:
    The chanallanged being solved is creating a reinforcement learning agent that can balance a pole on a cart. Without direct instruction the agent must learn to move the pole left or right at the 
    correct times to keep the pole balanced. It must learn this through trial and error, receiving rewards for successful balancing and penalties for failures. Part of the difficulty lies in the amount of reward and penalty the agent receives, as well as the complexity of the environment itself.
    given for each action. In total we want to keep the pole upright for 200 steps to consider the environment solved.

Understanding the type and nature of RL to be used:
    This model uses unsupervised reinforcement learning with a trial and error approach. The agent is given a very basic policy of, if the pole is leaning left, move left if its leaning right move right.
    This decision is then returned back the the model and it returns a reward or penalty based on the action taken. 
    Only a basic policy hsa been implimented, so the model is not actually learning here, it is simply just following a
    set of instructions. This is why you see the model can only hold hold the pole up for 63 steps, well short of the 200.
    
    Later in the notebook a more complex policy is implimented using Q-learning, 
    which allows the model to learn from its mistakes and improve over time. A Deep Q-Network
    is intitalized to allow the model to learn a better policy from scratch by interacting with the envirnemnt
    then using the rewards and penalties to update its Q-values and improve its decision-making process.
    
    The policy gradient section is used to make good descions more probable and bad descions less probable. 
    This is the renforcement part of RL learning, the algorithm renforces good descisions by making them more likely
    to be chosen in the future. This is done by the discount and normalize reawrds function, which makes it so that
    early rewards are the sum off all rewards that came before it, but future rewards are discounted by a set rate
    to allow the model to correctly correlate its sucsess with its earlier actions. They are also normalized to 
    add stability to the learning process.

    
    We a that the model is initalized with an input layer that takes in the 4 state variables
    Along with that a "replay buffer" which is basically an array with previous experinces that the model
    learns from a randomly sampled bunch to endssure it is not overfitting.
    The model is then trained in a loop of 600 "episodes" (trials) where it plays a full attempt of the game.
    In each episode, it uses a epsilon-greedy policy, so that it balances learning new decisions and making the best decision
    Basically, either the model makes a random decision or it makes the best decision based on its current knowledge.
    The tuple that stores the information about that step is then returned to the replay_buffer, and if the buffer is
    large enough then the model samples a mini-batch of experiences to learn from. The model then updates its Q-values
    based on the rewards and penalties received.
    
    The last 3 versions of the DQN all work on the problem of the model having to do 2 things at once
    it has to predict the Q-values for the current state and action as well as for the next state. This means that
    as the model adjusts its weights, the target q-values also shift. The Double DQN, Dueling Double DQN and
    Fixed Q-Value Targets all attempt to solve this problem in different ways, this the goal of stabilizing the convergence
    of the model to an optimal policy.

"""

#My attempt at improving the Trial and Error Policy
#Most of the code is the original code from the notebook, just with a modified basic policy function
import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=42)

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

#Code taken Directly from given notebook
def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
for episode in range(500):
    episode_rewards = 0
    obs, info = env.reset(seed=episode)
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_rewards += reward
        if done or truncated:
            break

    totals.append(episode_rewards)
print("--- Basic Policy ---")
print(f"Mean total rewards over 500 episodes: {np.mean(totals)}")
print(f"Standard deviation: {np.std(totals)}")
print(f"Minimum rewards in an episode: {np.min(totals)}")
print(f"Maximum rewards in an episode: {np.max(totals)}")

#With the basic policy, we get a Maximum Reward of 63 and a minimum of 22, My changes (Documented below)

def my_policy(obs):
    """
   By adding the angular velocity to the angle, we can tell if the momentum of the pole 
   is pushing it away from where its leaning or towards this. If its still moving in that direction we 
   can push the cart in that direction to counteract the movement and help balance the pole.
    """
    angle = obs[2]
    angular_velocity = obs[3]
    if angle + angular_velocity < 0: #Because right is the positive direction, if the sum is negative its falling left
        return 0  # left
    else:
        return 1  # right #If its not falling to the left, it is falling right 99.99999% of the time
    
    
#This code is directly taken from the given notebook
totals = []
for episode in range(500):
    episode_rewards = 0
    obs, info = env.reset(seed=episode)
    for step in range(200):
        action = my_policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_rewards += reward
        if done or truncated:
            break

    totals.append(episode_rewards)
print("--- My Policy ---")
print(f"Mean total rewards over 500 episodes: {np.mean(totals)}")
print(f"Standard deviation: {np.std(totals)}")
print(f"Minimum rewards in an episode: {np.min(totals)}")
print(f"Maximum rewards in an episode: {np.max(totals)}")

# Now the model passes the enviroment with a max reward of 200