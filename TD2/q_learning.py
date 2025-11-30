import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """
    best_next = np.max(Q[sprime])
    Q[s, a] = Q[s, a] + alpha * (r + gamma * best_next - Q[s, a])
    return Q

def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    Takes as unput the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
    if np.random.rand() < epsilone:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[s]) 

if __name__ == "__main__":
    env = gym.make("Taxi-v3")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.01 # Learning rate

    gamma = 0.99 # Discount factor

    epsilon = 1.0 # Exploration rate for espilon-greedy policy

    n_epochs = 20000 # choose your own
    max_itr_per_epoch = 200 # choose your own
    rewards = []

    for e in range(n_epochs):
        r = 0

        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            # Update state and put a stoping criteria
            S = Sprime  # mise à jour de l’état

            if done:
                break

        rewards.append(r)

        epsilon = max(0.05, epsilon * 0.999)

        if e % 1000 == 0:
            print(f"Episode {e}, reward = {r}, epsilon = {epsilon:.3f}")

    # plot the rewards in function of epochs

    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Q-learning on Taxi-v3")
    plt.show()

    print("Training finished \n")
    
    """
    
    Evaluate the q-learning algorihtm
    
    """

    env.close()
