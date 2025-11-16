import numpy as np

int_to_char = {
    0 : 'u',
    1 : 'r',
    2 : 'd',
    3 : 'l'
}

policy_one_step_look_ahead = {
    0 : [-1,0],  # up
    1 : [0,1],   # right
    2 : [1,0],   # down
    3 : [0,-1]   # left
}

def policy_int_to_char(pi,n):

    pi_char = ['']

    for i in range(n):
        for j in range(n):

            if i == 0 and j == 0 or i == n-1 and j == n-1:
                continue

            pi_char.append(int_to_char[pi[i,j]])

    pi_char.append('')
    return np.asarray(pi_char).reshape(n,n)



def policy_evaluation(n, pi, v, Gamma, threshhold):
    """
    Évalue V^pi : on applique Bellman expectation jusqu'à convergence.
    """

    while True:
        delta = 0
        new_v = np.copy(v)

        for i in range(n):
            for j in range(n):

                # final states
                if (i == 0 and j == 0) or (i == n-1 and j == n-1):
                    new_v[i, j] = 0
                    continue

                a = pi[i, j]
                di, dj = policy_one_step_look_ahead[a]

                # next state
                ni, nj = i + di, j + dj

                # if out of grid - doesn't move
                if ni < 0 or ni >= n or nj < 0 or nj >= n:
                    ni, nj = i, j

                reward = -1

                new_v[i, j] = reward + Gamma * v[ni, nj]

                delta = max(delta, abs(new_v[i,j] - v[i,j]))

        v = new_v

        if delta < threshhold:
            break

    return v



def policy_improvement(n, pi, v, Gamma):
    new_pi = np.copy(pi)
    policy_stable = True

    for i in range(n):
        for j in range(n):

            # final states
            if (i == 0 and j == 0) or (i == n-1 and j == n-1):
                continue

            old_action = pi[i, j]

            # Optimal actions in this state
            values = []
            for a in range(4):
                di, dj = policy_one_step_look_ahead[a]
                ni, nj = i + di, j + dj

                if ni < 0 or ni >= n or nj < 0 or nj >= n:
                    ni, nj = i, j

                reward = -1
                values.append(reward + Gamma * v[ni, nj])

            best_action = np.argmax(values)
            new_pi[i, j] = best_action

            if best_action != old_action:
                policy_stable = False

    return new_pi, policy_stable



def policy_initialization(n):
    """
    Politique aléatoire (chaque état non terminal choisit une action au hasard)
    """
    pi = np.random.randint(low=0, high=4, size=(n,n))

    # final states
    pi[0,0] = 0
    pi[n-1,n-1] = 0

    return pi



def policy_iteration(n, Gamma, threshhold):

    pi = policy_initialization(n=n)
    v = np.zeros((n,n))

    while True:
        v = policy_evaluation(n=n, v=v, pi=pi, threshhold=threshhold, Gamma=Gamma)
        pi, pi_stable = policy_improvement(n=n, pi=pi, v=v, Gamma=Gamma)

        if pi_stable:
            break

    return pi, v



n = 4
Gamma = [0.8, 0.9, 1.0]
threshhold = 1e-4

for _gamma in Gamma:
    pi, v = policy_iteration(n=n, Gamma=_gamma, threshhold=threshhold)
    pi_char = policy_int_to_char(pi=pi, n=n)

    print("\nGamma =", _gamma, "\n")
    print(pi_char, "\n")
    print(v)
