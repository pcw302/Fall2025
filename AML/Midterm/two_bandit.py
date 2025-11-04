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
        self.__known_best_epsilon = None  # This is only here to be able to highlight the best epsilon on the plot


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
        

    def run_simulation(self, printing: bool = False) -> None:
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

            if printing:
                print("---Greedy Policy---")
                print(f"Total pulls: {pulls_greedy}")
                print(f"Total rewards: {sum_rewards_greedy}")
                print(f"Average rewards: {self.average_rewards_greedy}")
                print("\n---Epsilon-Greedy Policy---")
                print(f"Total pulls: {pulls_epsilon}")
                print(f"Total rewards: {sum_rewards_epsilon}")
                print(f"Average rewards: {self.average_rewards_epsilon}")
                
    def run_batch_simulation(self, batches: int = 100, printing: bool = False) -> None:
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
        if printing:
            print("\n---Averaged Results over Batches---")
            print("---Greedy Policy---")
            print(f"Average pulls: {total_pulls_greedy / batches}")
            print(f"Average rewards: {total_rewards_greedy / batches}")
            print("\n---Epsilon-Greedy Policy---")
            print(f"Average pulls: {total_pulls_epsilon / batches}")
            print(f"Average rewards: {total_rewards_epsilon / batches}")
        else:
            return


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
        
        
    def optimize_epsilon(self, epsilon_values: list[float], plot: bool = True, inplace: bool = False) -> float:
        """ ---Finds the best epsilon value from a list and plots the results--- \n
        inputs:
            epsilon_values: list of float, epsilon values to test
            plot: bool, whether to plot the results. True by default.
            inplace: bool, whether to update the object's epsilon value. False by default.
        outputs:
            float, best epsilon value
        """
        best_epsilon = epsilon_values[0]
        best_average_reward = -1
        average_rewards_per_epsilon = []

        for epsilon in epsilon_values:
            self.epsilon = epsilon
            self.reset()
            # Run a simulation without printing results
            self.run_simulation(printing=False) 
            
            # Calculate total rewards for this epsilon
            total_rewards = sum(self.reward_history_epsilon[0]) + sum(self.reward_history_epsilon[1])
            average_reward = total_rewards / self.episodes
            average_rewards_per_epsilon.append(average_reward)

            if average_reward > best_average_reward:
                best_average_reward = average_reward
                best_epsilon = epsilon
        self.__known_best_epsilon = best_epsilon  # Store for plot
        if plot:
            self._plot_epsilon_optimization(epsilon_values, average_rewards_per_epsilon)
        if inplace:
            self.epsilon = best_epsilon

        print(f"Best epsilon found: {best_epsilon:.3f} with an average reward of {best_average_reward:.3f}")
        return best_epsilon

    def _run_single_simulation(self, policy_name: str) -> np.ndarray:
            """Helper to run one simulation for a single policy and return reward history."""
            
            local_avg_rewards = [0.0, 0.0] 
            pulls = [0, 0]
            sum_rewards = [0, 0]
            temp_reward_history = np.zeros(self.episodes)
            
            for i in range(self.episodes):
                
                # 1. Choose Arm
                if policy_name == 'greedy':
                    choice = i % 2 if i < 2 else np.argmax(local_avg_rewards) # Choose both arms once first.
                    
                    
                elif policy_name == 'epsilon':
                    if np.random.rand() < self.epsilon:
                        choice = np.random.choice([0, 1])
                    else:
                        choice = np.argmax(local_avg_rewards) #[Arm 1, Arm 2]
                else:
                    raise ValueError("Invalid policy_name")

                reward = self.pull_arm(choice)
                
                # 3. Update Statistics
                pulls[choice] += 1
                sum_rewards[choice] += reward
                local_avg_rewards[choice] = sum_rewards[choice] / pulls[choice]
                temp_reward_history[i] = reward
                
            return temp_reward_history


    def plot_results(self) -> None:
        """
        Runs multiple simulations for each policy and plots the average reward per step over time.
        inputs:
            runs: int, The number of independent simulations to average over.
        outputs:
            None, displays a plot comparing the learning curves of the two policies.
        """
        import matplotlib.pyplot as plt

        # Arrays to store the reward at each step for every run
        temp_greedy_rewards = np.zeros((self.episodes, self.episodes))
        temp_epsilon_rewards = np.zeros((self.episodes, self.episodes))

        for i in range(self.episodes):
            temp_greedy_rewards[i, :] = self._run_single_simulation('greedy')
            temp_epsilon_rewards[i, :] = self._run_single_simulation('epsilon')

        mean_greedy_rewards = np.mean(temp_greedy_rewards, axis=0)
        mean_epsilon_rewards = np.mean(temp_epsilon_rewards, axis=0)

        plt.figure(figsize=(12, 8))
        plt.title("Policy Performance Comparison over Time")
        plt.plot(np.arange(self.episodes), mean_greedy_rewards, label="Greedy Policy")
        plt.plot(np.arange(self.episodes), mean_epsilon_rewards, label=f"Epsilon-Greedy Policy (ε={self.epsilon})")
        plt.xlabel("Episodes (Arm Pulls)")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
    def _plot_epsilon_optimization(self, epsilon_values: list[float], average_rewards: list[float]) -> None:
        """ ---Helper, Use optimize_epsilon(plot = True)--- \n
        inputs:
            epsilon_values: list of float, epsilon values tested
            average_rewards: list of float, corresponding average rewards
        outputs:
            None, displays plot
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.title("Epsilon Optimization: Average Rewards vs Epsilon Values")
        plt.scatter(epsilon_values, average_rewards,
                 marker='o', 
                 alpha=0.7, 
                 color='slategray',
                 s=3,)
        plt.scatter(self.__known_best_epsilon, max(average_rewards),
                    marker='*',
                    color='red',
                    alpha=0.9,
                    s=100,
                    label=f'Best Epsilon: {self.__known_best_epsilon:.3f}')
        plt.xlabel('Epsilon Values')
        plt.ylabel('Average Rewards')
        plt.grid()
        plt.show()

    def change_epsilon(self, new_epsilon: float) -> None:
        """ ---Changes the epsilon value--- \n
        inputs:
            new_epsilon: float, new epsilon value
        outputs:
            None
        """
        if 0 < new_epsilon < 1:
            self.epsilon = new_epsilon
        else:
            raise ValueError("Epsilon must be between 0 and 1.")

test = TwoArmedBandit(epsilon=0.01, episodes=5000)
#test.run_simulation()
#test.run_batch_simulation(batches=100)
#test.optimize_epsilon(epsilon_values=np.linspace(0.01, 0.9, 1000).tolist(), plot=False, inplace=True)
test.plot_results()