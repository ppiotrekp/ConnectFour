import numpy as np
import random
import matplotlib.pyplot as plt

class ConnectFour:
    def __init__(self, rows=6, columns=7):
        self.rows = rows
        self.columns = columns
        self.board = np.zeros((rows, columns))

    def reset(self):
        self.board = np.zeros((self.rows, self.columns))
        return self.board

    def is_valid_move(self, action):
        return self.board[0, action] == 0

    def get_valid_moves(self):
        return [c for c in range(self.columns) if self.is_valid_move(c)]

    def step(self, action, player):
        if not self.is_valid_move(action):
            return self.board, -1, True  # invalid move
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, action] == 0:
                self.board[r, action] = player
                break
        win = self.check_win(player)
        done = win or len(self.get_valid_moves()) == 0
        reward = 1 if win else 0
        return self.board, reward, done

    def check_win(self, player):
        for r in range(self.rows):
            for c in range(self.columns - 3):
                if self.board[r, c] == self.board[r, c+1] == self.board[r, c+2] == self.board[r, c+3] == player:
                    return True
        for r in range(self.rows - 3):
            for c in range(self.columns):
                if self.board[r, c] == self.board[r+1, c] == self.board[r+2, c] == self.board[r+3, c] == player:
                    return True
        for r in range(self.rows - 3):
            for c in range(self.columns - 3):
                if self.board[r, c] == self.board[r+1, c+1] == self.board[r+2, c+2] == self.board[r+3, c+3] == player:
                    return True
        for r in range(3, self.rows):
            for c in range(self.columns - 3):
                if self.board[r, c] == self.board[r-1, c+1] == self.board[r-2, c+2] == self.board[r-3, c+3] == player:
                    return True
        return False
    def render(self):
        print("  ".join(map(str, range(self.columns))))
        print("\n".join(["  ".join(map(str, row)) for row in self.board.astype(int)]))
        print()

class QLearningAgent:
    def __init__(self, actions, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.q_table = {}  # state-action pairs

    def get_action(self, state):
        state = tuple(state.flatten())
        if random.uniform(0, 1) < self.epsilon or state not in self.q_table:
            return random.choice(self.actions)  # explore
        else:
            return np.argmax(self.q_table[state])  # exploit

    def update_q_table(self, state, action, reward, next_state):
        state = tuple(state.flatten())
        next_state = tuple(next_state.flatten())
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))

        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

def train_agent(agent, game, episodes):
    wins = []  # List to store the number of wins
    for episode in range(episodes):
        state = game.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = game.step(action, 1)
            if not done:  # if game is not finished, opponent makes a move
                next_state, _, done = game.step(random.choice(game.get_valid_moves()), -1)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
        wins.append(reward)  # Store the result of the game
        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Q-table size: {len(agent.q_table)}")
    return wins

def simulate_game(agent1, agent2, game):
    state = game.reset()
    game.render()
    done = False
    winner = None
    while not done:
        action1 = agent1.get_action(state)
        _, _, done = game.step(action1, 1)
        game.render()
        if done:
            winner = 'Agent 1'
            break
        action2 = agent2.get_action(state)
        _, _, done = game.step(action2, 2)
        game.render()
        if done:
            winner = 'Agent 2'
    print(f"The winner is: {winner}")
    return game.board

game = ConnectFour()

agent = QLearningAgent(actions=list(range(game.columns)))
game = ConnectFour()
results = train_agent(agent, game, 50)

agent1 = QLearningAgent(game.get_valid_moves())
agent2 = QLearningAgent(game.get_valid_moves())

train_agent(agent1, game, 50)
train_agent(agent2, game, 50)

simulate_game(agent1, agent2, game)

plt.plot(results)
plt.xlabel('Episode')
plt.ylabel('Wins')
plt.show()