import numpy as np
import os, sys

def generate_location(w, h):
    x = np.random.random_integers(w)-1
    y = np.random.random_integers(h)-1
    return (x, y)

class Type:
    empty = 0
    food = 1
    wall = 2
    snake = 3

class Actions:
    left = 0
    right = 1
    straight = 2
    __length__ = 3

class Directions:
    left = 0
    right = 1
    up = 2
    down = 3
    __length__ = 4

def clip_position(pt, x_range, y_range):
    x = max(pt[0], x_range[0])
    x = min(pt[0], x_range[1])
    y = max(pt[1], y_range[0])
    y = min(pt[1], y_range[1])

    return (x, y)

class Environment:
    def __init__(self):
        # Initialize the environment
        self.width = 100
        self.height = 100
        self.config = np.zeros((self.height, self.width)) + Type.empty

        self.x_range = (0, self.width)
        self.y_range = (0, self.height)

        self.generate_food()

        # Generate a random (x, y) - snake_head
        self.snake_head = generate_location(self.width, self.height)
        self.snake_dir = Directions.random()
        self.config[self.snake_head] = Type.snake

        self.delta_pos_head = [
            [
                [0, +1], [0, -1], [-1, 0]
            ],
            [
                [0, -1], [0, +1], [+1, 0]
            ],
            [
                [-1, 0], [+1, 0], [0, -1]
            ],
            [
                [+1, 0], [-1, 0], [0, +1]
            ]
        ]

        # No walls for now
        # Also, snake occupies one square and doesn't grow

    def get_reward(self, state):
        head = state.head
        if self.config[head] == Type.empty:
            return 0
        elif self.config[head] == Type.food:
            return 10

    def update_config(self, state, next_state):
        head = state.head
        next_head = next_state.head
        self.config[head] = Type.empty
        if self.config[next_head] == Type.food:
            self.generate_food()
        self.config[next_head] = Type.empty

    def generate_food(self):
        # Generate a random (x, y) - food
        self.food = generate_location(self.width, self.height)
        self.config[self.food] = Type.food

    def submit_action(self, state, action):
        # state.head --> snake's position
        # state.dir --> snake's direction
        # action --> action that is taken
        del_head = self.delta_pos_head[(state.dir, action)]
        head = clip_position(state.head + del_head, self.x_range, self.y_range)

        next_state = State()
        next_state.head = head
        next_state.dir = action

        reward = self.get_reward(next_state)
        update_config(state, next_state)

        return next_state, reward

def State():
    pass

def Agent():)
    def __init__():
        env = Environment()
