import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# hyperparams 
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
NUM_EPISODES = 10_000
TRAIN_SEED = 42
EVAL_SEED = 123
ACTIONS = ("Stick", "Hit")

# environment
env = gym.make("Blackjack-v1", natural=False, sab=True)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# q-table - sum, dealer, ace, actions
q_table = np.zeros((32, 11, 2, 2))
print("Q-table shape:", q_table.shape)

# training setup
rng = np.random.default_rng(TRAIN_SEED)
epsilon = EPSILON_START
episode_returns = []

state, info = env.reset(seed=TRAIN_SEED)

# training progress
for episode in range(NUM_EPISODES):
    total_reward = 0
    done = False

    while not done:
        # epsilon greedy approach
        if rng.random() < epsilon:
            action = int(rng.integers(2))
        else:
            action = int(np.argmax(q_table[state]))

        next_state, reward, terminated, truncated, info = env.step(action)

        # no future value once the hand is over
        if terminated:
            target = reward
        else:
            target = reward + GAMMA * np.max(q_table[next_state])

        q_table[state][action] += ALPHA * (target - q_table[state][action])

        state = next_state
        total_reward += reward
        done = terminated or truncated

    episode_returns.append(total_reward)

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # progress every 1000 
    if (episode + 1) % 1000 == 0:
        print("Episode", episode + 1, "| epsilon:", round(epsilon, 3), "| mean return (last 1000):", np.mean(episode_returns[-1000:]))

    state, info = env.reset()

env.close()

# evaluation setup
eval_env = gym.make("Blackjack-v1", natural=False, sab=True)
eval_returns = []

state, info = eval_env.reset(seed=EVAL_SEED)

# evaluation progress
for episode in range(NUM_EPISODES):
    total_reward = 0
    done = False

    while not done:
        action = int(np.argmax(q_table[state]))
        state, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    eval_returns.append(total_reward)
    state, info = eval_env.reset()

eval_env.close()
eval_returns = np.array(eval_returns)

# printing results
results = (
    "Evaluation: frozen greedy, ties go to stick\n"
    f"Training seed: {TRAIN_SEED} | Eval seed: {EVAL_SEED}\n"
    f"Episodes: {NUM_EPISODES}\n"
    f"Mean return: {eval_returns.mean():.4f}\n"
    f"Win rate: {np.mean(eval_returns > 0):.2%}\n"
    f"Draw rate: {np.mean(eval_returns == 0):.2%}\n"
    f"Loss rate: {np.mean(eval_returns < 0):.2%}\n"
)
print("\n" + results)

os.makedirs("results/", exist_ok=True)
with open("results/results.txt", "w") as f:
    f.write(results)


# training graph
window = 500
moving_average = np.convolve(episode_returns, np.ones(window) / window, mode="valid")

plt.figure(figsize=(9, 5))
plt.plot(np.arange(window, NUM_EPISODES + 1), moving_average)
plt.xlabel("Episode")
plt.ylabel("Average return (last 500 episodes)")
plt.title("Blackjack Q-Learning Training Progress")
plt.grid(True)
plt.savefig("results/training_graph.png")
plt.close()


# q-table graph
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for usable_ace in range(2):
    for action in range(2):
        ax = axes[usable_ace][action]

        # player sums 4-21 and dealer cards 1-10
        image = ax.imshow(
            q_table[4:22, 1:11, usable_ace, action],
            origin="lower",
            extent=[0.5, 10.5, 3.5, 21.5],
            aspect="auto",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )

        ax.set_title(
            f"{ACTIONS[action]} (action {action}), usable ace = {usable_ace}"
        )
        ax.set_xlabel("Dealer showing card (1 = ace)")
        ax.set_ylabel("Player sum")
        ax.set_xticks(range(1, 11))
        ax.set_yticks(range(4, 22))
        fig.colorbar(image, ax=ax, label="Q-value")

fig.suptitle("Blackjack Q-table")
fig.tight_layout()
fig.savefig("results/q_table.png")
plt.close(fig)
