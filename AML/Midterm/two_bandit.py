import numpy as np

# For this file, we are using a two-armed bandit environment, where one machine has a higher probability of payout than the other.
# First we will test the greedy policy, then we will implement an epsilon-greedy policy to see if it performs better.
class TwoArmedBandit:
    def __init__(self, epsilon: float = 0.1, episodes: int = 5000):
        """ ---Initializes the Two-Armed Bandit--- \n
        inputs:
            epsilon: float, probability of exploration for epsilon-greedy policy
            episodes: int, number of times to pull an arm
        outputs:
            None
        """
        
        self.reward_history_greedy = {0: [], 1: []}  # Store rewards for each arm
        self.reward_history_epsilon = {0: [], 1: []}  # Store rewards for each arm
        self.choice_history_greedy = {0: [], 1: []}  # Store choices for each arm
        self.choice_history_epsilon = {0: [], 1: []}  # Store choices for each arm
        self.average_rewards_greedy = [0.0, 0.0]  # Average rewards for each arm (Greedy)
        self.average_rewards_epsilon = [0.0, 0.0]  # Average rewards for each arm (Epsilon-Greedy)
        self.epsilon = epsilon if epsilon is not None and 0 < epsilon < 1 else 0.1  # Exploration probability for epsilon-greedy policy
        self.true_arm_one = 0.8 #Good arm
        self.true_arm_two = 0.2 #Bad arm
        self.episodes = episodes


    def pull_arm(self, arm: int) -> 0 | 1:
        """ ---Simulates the pulling of an arm--- \n
        inputs:
            arm: float, probability of payout for the selected arm
        outputs:
            int, 1 for payout, 0 for no payout
        """
        
        match arm:
            case 0:
                return 1 if np.random.rand() < self.true_arm_one else 0 #Good arm
            case 1:
                return 1 if np.random.rand() < self.true_arm_two else 0 #Bad arm

    def greedy_policy(self) -> 0 | 1:
        """ ---Greedy Policy--- \n
        Will always choose the arm with the highest average reward so far.
        """
        return np.argmax(self.average_rewards_greedy) #Will either return 0 for arm 1 of 1 for arm 2

    def epsilon_greedy_policy(self, epsilon=0.1) -> 0 | 1:
        """ ---Epsilon-Greedy Policy--- \n
        With probability epsilon, will choose a random arm.
        With probability 1-epsilon, will choose the arm with the highest average reward so far.
        """
        if np.random.rand() < epsilon:
            return np.random.choice([0, 1]) #Randomly choose an arm
        else:
            return np.argmax(self.average_rewards_epsilon) #Choose the best arm so far
        

    def run_simulation(self) -> None:
            """ ---Runs the simulation for both policies--- \n
            inputs:
                episodes: int, number of times to pull an arm
            outputs:
                None, prints the results of the simulation
            """
            
            pulls_greedy = [0, 0]  # Number of times each arm has been pulled [Arm 1 (Good), Arm 2 (Bad)]
            pulls_epsilon = [0, 0] 
            sum_rewards_greedy = [0, 0]  # Total rewards for each arm
            sum_rewards_epsilon = [0, 0]

            for i in range(self.episodes):
                #Greedy Block
                if i == 0:
                    choice = np.random.choice([0, 1])  # Randomly choose an arm for the first pull
                else:
                    choice = self.greedy_policy()
                reward = self.pull_arm(choice)
                pulls_greedy[choice] += 1
                sum_rewards_greedy[choice] += reward
                self.average_rewards_greedy[choice] = sum_rewards_greedy[choice] / pulls_greedy[choice]
                self.reward_history_greedy[choice].append(reward)
                self.choice_history_greedy[choice].append(choice)

                #Epsilon-Greedy Block
                if i == 0:
                    choice = np.random.choice([0, 1])  # Randomly choose an arm for the first pull
                else:
                    choice = self.epsilon_greedy_policy(epsilon=self.epsilon)
                reward = self.pull_arm(choice)
                pulls_epsilon[choice] += 1
                sum_rewards_epsilon[choice] += reward
                self.average_rewards_epsilon[choice] = sum_rewards_epsilon[choice] / pulls_epsilon[choice]
                self.reward_history_epsilon[choice].append(reward)
                self.choice_history_epsilon[choice].append(choice)

            print("---Greedy Policy---")
            print(f"Total pulls: {pulls_greedy}")
            print(f"Total rewards: {sum_rewards_greedy}")
            print(f"Average rewards: {self.average_rewards_greedy}")
            print("\n---Epsilon-Greedy Policy---")
            print(f"Total pulls: {pulls_epsilon}")
            print(f"Total rewards: {sum_rewards_epsilon}")
            print(f"Average rewards: {self.average_rewards_epsilon}")
                
    def run_batch_simulation(self, batches: int = 100) -> None:
        """ ---Runs multiple simulations and averages the results--- \n
        inputs:
            batches: int, number of simulations to run
        outputs:
            None, prints the averaged results of the simulations
        """
        total_pulls_greedy = np.array([0, 0])
        total_rewards_greedy = np.array([0, 0])
        total_pulls_epsilon = np.array([0, 0])
        total_rewards_epsilon = np.array([0, 0])

        for _ in range(batches):
            self.run_simulation()
            total_pulls_greedy += np.array([len(self.choice_history_greedy[0]), len(self.choice_history_greedy[1])])
            total_rewards_greedy += np.array([sum(self.reward_history_greedy[0]), sum(self.reward_history_greedy[1])])
            total_pulls_epsilon += np.array([len(self.choice_history_epsilon[0]), len(self.choice_history_epsilon[1])])
            total_rewards_epsilon += np.array([sum(self.reward_history_epsilon[0]), sum(self.reward_history_epsilon[1])])

        print("\n---Averaged Results over Batches---")
        print("---Greedy Policy---")
        print(f"Average pulls: {total_pulls_greedy / batches}")
        print(f"Average rewards: {total_rewards_greedy / batches}")
        print("\n---Epsilon-Greedy Policy---")
        print(f"Average pulls: {total_pulls_epsilon / batches}")
        print(f"Average rewards: {total_rewards_epsilon / batches}")


    def reset(self) -> None:
        """ ---Resets the bandit to initial state--- \n
        inputs:
            None
        outputs:
            None
        """
        self.reward_history_greedy = {0: [], 1: []}
        self.reward_history_epsilon = {0: [], 1: []}
        self.choice_history_greedy = {0: [], 1: []}
        self.choice_history_epsilon = {0: [], 1: []}
        self.average_rewards_greedy = [0.0, 0.0]
        self.average_rewards_epsilon = [0.0, 0.0]
        
        
    def optimize_epsilon(self, epsilon_values: list[float]) -> float:
        """ ---Finds the best epsilon value from a list--- \n
        inputs:
            epsilon_values: list of float, epsilon values to test
        outputs:
            float, best epsilon value
        """
        best_epsilon = epsilon_values[0]
        best_average_reward = -1

        for epsilon in epsilon_values:
            self.epsilon = epsilon
            self.reset()
            self.run_simulation()
            total_rewards = sum(self.reward_history_epsilon[0]) + sum(self.reward_history_epsilon[1])
            average_reward = total_rewards / self.episodes

            if average_reward > best_average_reward:
                best_average_reward = average_reward
                best_epsilon = epsilon

        print(f"Best epsilon: {best_epsilon} with average reward: {best_average_reward}")
        return best_epsilon

    def plot_results(self) -> None:
        """ ---Plots the results of the simulation--- \n
        inputs:
            None
        outputs:
            None, displays plots
        """
        import matplotlib.pyplot as plt

        # Plot for Greedy Policy
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title("Greedy Policy: Cumulative Rewards Over Time")
        cumulative_rewards_greedy = np.cumsum([sum(self.reward_history_greedy[0]), sum(self.reward_history_greedy[1])])
        plt.plot(cumulative_rewards_greedy, label='Cumulative Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Rewards')
        plt.legend()

        # Plot for Epsilon-Greedy Policy
        plt.subplot(1, 2, 2)
        plt.title("Epsilon-Greedy Policy: Cumulative Rewards Over Time")
        cumulative_rewards_epsilon = np.cumsum([sum(self.reward_history_epsilon[0]), sum(self.reward_history_epsilon[1])])
        plt.plot(cumulative_rewards_epsilon, label='Cumulative Rewards', color='orange')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Rewards')
        plt.legend()

        plt.tight_layout()
        plt.show()

test = TwoArmedBandit(epsilon=0.1, episodes=5000)
#test.run_simulation()
test.run_batch_simulation(batches=100)
test.reset()
test.optimize_epsilon(epsilon_values=np.linspace(0.01, 0.9, 100).tolist())