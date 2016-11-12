import numpy as np
import os, sys

import time
import cv2

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

    @classmethod
    def random(self):
        return np.random.random_integers(self.__length__) - 1

class Directions:
    left = 0
    right = 1
    up = 2
    down = 3
    __length__ = 4

    @classmethod
    def random(self):
        return np.random.random_integers(self.__length__) - 1

directions = ['left', 'right', 'up', 'down']
actions = ['left', 'right', 'straight']
# colors = [(0, 0, 0), (255, 255, 0), (0, 255, 255), (255, 255, 255)]
colors = [0, 127, 127, 255]

class State:
    def __str__(self):
        return str(self.head) + ' ' + directions[self.dir]

def generate_location(w, h):
    x = np.random.random_integers(w)-1
    y = np.random.random_integers(h)-1
    return (x, y)

def clip_position(pt, x_range, y_range):
    x, y = pt
    x = max(x, x_range[0])
    x = min(x, x_range[1])
    y = max(y, y_range[0])
    y = min(y, y_range[1])

    return (x, y)

class Environment:
    def __init__(self):
        # Initialize the environment
        self.width = 10
        self.height = 10
        self.config = np.zeros((self.height, self.width)) + Type.empty

        self.x_range = (0, self.width - 1)
        self.y_range = (0, self.height - 1)

        self.win = False

        self.delta_pos_head = np.array([
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
        ])

        self.next_direction = np.array([
            [
                Directions.down, Directions.up, Directions.left
            ],
            [
                Directions.up, Directions.down, Directions.right
            ],
            [
                Directions.left, Directions.right, Directions.up
            ],
            [
                Directions.right, Directions.left, Directions.down
            ]
        ])

    def init(self):
        self.generate_food()

        # Generate a random (x, y) - snake_head
        snake_head = generate_location(self.width, self.height)
        snake_dir = Directions.random()
        self.config[snake_head] = Type.snake

        # No walls for now
        # Also, snake occupies one square and doesn't grow

        init_state = State()
        init_state.head = snake_head
        init_state.dir = snake_dir
        init_state.food = self.food

        return init_state

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
        self.config[next_head] = Type.snake

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
        # print del_head, state.head, head

        next_state = State()
        next_state.head = head
        next_state.dir = self.next_direction[(state.dir, action)]

        reward = self.get_reward(next_state)
        self.update_config(state, next_state)

        next_state.food = self.food

        return next_state, reward

    def visualise(self):
        self.win = True
        mul = 10
        shape_ = self.config.shape
        shape = tuple(np.array(list(shape_))*mul)
        img = np.zeros(shape)

        for j in range(0, shape_[0]):
            for i in range(0, shape_[1]):
                img[i*mul:(i+1)*mul, j*mul:(j+1)*mul] = np.zeros((mul, mul)) + colors[int(self.config[i, j])]

        # print img
        img = np.uint8(img)
        cv2.imshow("win", img)
        cv2.waitKey(1)

class Agent:
    def __init__(self):
        self.env = Environment()
        self.state = self.env.init()

        self.FEATURE_LENGTH = 6

    def log(self):
        print self.state, actions[action], reward, next_state

    def extract_features(self, state):
        return np.array(np.sign(np.array(state.head) + np.array(state.food)).tolist() + [state.dir])

    def init_weights(self):
        self.Q = Counter()
        self.weights = np.random.random(self.FEATURE_LENGTH)

    def update_weights(self):


    def learn(self):
        for i in range(100):
            action = Actions.random()
            next_state, reward = self.env.submit_action(self.state, action)

            self.log()

            # Train the RL agent

            self.env.visualise()
            self.state = next_state

            time.sleep(0.1)

agent = Agent()
agent.learn()
