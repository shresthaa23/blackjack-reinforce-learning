"""Install once:  python -m pip install gymnasium
Run:           python frozenlake_demo.py

Try changing the moves below and predicting the result before running.
"""

import gymnasium as gym


# 1. MAKE: create a game.
env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=False,
    render_mode="ansi",
    max_episode_steps=10,
)

print("FROZENLAKE: reach G without falling into H")
print("Map: S = start, F = safe ice, H = hole, G = goal")
print("Observation: your square, numbered 0-15 from left to right, row by row")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Actions: 0 = left, 1 = down, 2 = right, 3 = up")
print("Rewards: 1 for reaching the goal; 0 for all other moves")
print("One step = one move. One episode = one game.")

action_names = ["left", "down", "right", "up"]

# Hand made moves.
demos = [
    ("Reach the goal", [2, 2, 1, 1, 1, 2]),
    ("Fall into a hole", [2, 1]),
    ("Reach the time limit", [0] * 10),
]

for title, moves in demos:
    print("\n" + "=" * 50)
    print(title)

    # 2. RESET: start a fresh episode. Returns two values.
    observation, info = env.reset(seed=42)
    print("Starting observation:", observation)
    print(env.render())

    for action in moves:
        # 3. STEP: take one action. Returns five values.
        observation, reward, terminated, truncated, info = env.step(action)

        print("Action:", action_names[action])
        print(env.render())
        print("Observation:", observation, "| Reward:", reward)
        print("Terminated:", terminated, "| Truncated:", truncated)
        print("Info (extra environment details):", info)

        # terminated: reached the goal or a hole.
        # truncated: stopped by our 10-step limit.
        # A reward of zero alone does NOT tell us whether the game ended.
        if terminated or truncated:
            if truncated:
                print("Episode ended: the step limit was reached.")
            elif reward == 1:
                print("Episode ended: you reached the goal!")
            else:
                print("Episode ended: you fell into a hole.")
            break  # Reset before taking any more actions.

# 4. CLOSE: release the environment's resources.
env.close()