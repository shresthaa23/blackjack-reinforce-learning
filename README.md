# blackjack-reinforce-learning
Training an artificial agent to play blackjack using reinforcement learning, q-learning. Uses Gymnasium's Blackjack-v1 with natural=False, sab=True. Trains the agent on 10000 episodes, then freezes the Q-table and evaluates the greedy policy on a new set of 10000 episodes. Prints results into `results/` folder with Win/Loss rates and average returns. Additionally, creates 2 visualizations, q-table heatmap and training graph. The heatmap shows the q-table for stick vs hit and with/without usable aces. The training graph is a 500 episode average of the return.

## To Run

```
uv venv
uv pip install -r requirements.txt
uv run <path-to-file>/project1.py
```

## Setup

| | |
|---|---|
| Python | 3.12.3 |
| gymnasium | 1.4.0 |
| numpy | 2.5.3 |
| matplotlib | 3.11.2 |
| Hardware | Intel Core i7-1360P laptop CPU (16 threads), 7.6 GB RAM, Ubuntu on WSL2, no GPU |
| Runtime | about 2 seconds end-to-end |

## Given Hyperparameters

| alpha | gamma | epsilon | epsilon_min | epsilon_decay | num_episodes |
|---|---|---|---|---|---|
| 0.1 | 0.99 | 1.0 | 0.01 | 0.995 | 10000 |

Epsilon is decayed once after each completed episode.

## Seeds and Tie-breaker

- Training seed - 42 
- Evaluation seed - 123 (new, separate environment to evaluate on)
- Tie-breaking: if Q values are equal, stick (action = 0) is chosen
- New state: Stick is chosen if a state is never seen before

## Evaluation Results

Frozen greedy policy on 10000 fresh episodes:

| Mean return | Win rate | Draw rate | Loss rate |
|---|---|---|---|
| -0.0677 | 42.22% | 8.79% | 48.99% |

## Q-Table Graph

This is a combination of 4 heatmaps since there are 4 potential dimensions for each case as seen in the table below. Each map plots the player's value vs the dealer's card. 

| | Left | Right |
|---|---|---|
| Top: no ace | Q-value of sticking | Q-value of hitting |
| Bottom: usable ace | Q-value of sticking | Q-value of hitting |

The color is the expected return of taking the action in the state with red being close to +1 a win, and blue -1 a loss. To make a decision, compare the same cell index of the 2 heatmaps in the same row and take the action with higher value or stick if equal.