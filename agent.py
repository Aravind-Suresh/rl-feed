import numpy as np

import os, time
import cv2
import cPickle as pickle

from util import *

"""
Consists of class definitions for Environment and Agent.
"""

class Environment:
    """
    Specifies the environment for the RL agent to interact.
    """
    DIM = 5
    MUL_FACTOR = 50
    # For visualisation, based on Type ( refer util.py )
    COLORS = [(0, 0, 0), (100, 255, 100), (255, 255, 255), (0, 0, 255)]

    def __init__(self, test = False, visualise = True):
        # Initialize the environment
        self.width = Environment.DIM
        self.height = Environment.DIM
        self.config = np.zeros((self.height, self.width), dtype = np.uint8) + Type.empty

        self.x_range = np.array([0, self.width - 1])
        self.y_range = np.array([0, self.height - 1])

        self.test = test

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
                Direction.down, Direction.up, Direction.left
            ],
            [
                Direction.up, Direction.down, Direction.right
            ],
            [
                Direction.left, Direction.right, Direction.up
            ],
            [
                Direction.right, Direction.left, Direction.down
            ]
        ])

        self.rewards = Params()
        self.rewards.win = 10
        self.rewards.live = 0

        self.show_visualisation = visualise

    def init(self):
        self.generate_food()

        # Generate a random (x, y) - snake_head
        snake_head = generate_location(self.x_range, self.y_range, [self.food])
        snake_dir = Direction.random()
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
        # print 121, self.config[head]
        if self.config[head] == Type.empty:
            return self.rewards.live
        elif self.config[head] == Type.food:
            return self.rewards.win
        else:
            return self.rewards.live

    def update_config(self, state, next_state):
        head = state.head
        next_head = next_state.head
        self.config[head] = Type.empty

        prev_type = self.config[next_head]
        self.config[next_head] = Type.snake

        if self.show_visualisation:
            self.visualise()

        if prev_type == Type.food:
            self.generate_food([next_head])

    def generate_food(self, arr = None):
        # Generate a random (x, y) - food
        x_range = self.x_range + [1, -1]
        y_range = self.y_range + [1, -1]

        self.food = generate_location(x_range, y_range, arr)
        self.config[self.food] = Type.food

    def submit_action(self, state, action):
        # state.head    -->     snake's position
        # state.dir     -->     snake's direction
        # action        -->     action that is taken

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
        mul = Environment.MUL_FACTOR
        # No. of channels
        ch = 3

        shape_ = self.config.shape
        shape = (shape_[0]*mul, shape_[1]*mul, ch)
        img = np.zeros(shape)

        for j in range(0, shape_[0]):
            for i in range(0, shape_[1]):
                img[i*mul:(i+1)*mul, j*mul:(j+1)*mul, :] = np.zeros((mul, mul, ch)) + Environment.COLORS[int(self.config[i, j])]

        # print img
        img = np.uint8(img)
        cv2.imshow("Fabber", img)
        cv2.waitKey(1)

class Agent:
    """
    The RL Agent class. Consists of methods for interacting with environment, and learning
    from the interaction. Supports loading/dumping learned model.
    """
    def __init__(self, input = None):
        self.env = Environment()
        self.state = self.env.init()

        self.params = Params()
        self.params.alpha = 0.5
        self.params.discount = 0.9
        self.params.iter = 10000

        self.initQ(input)

    def extract_features(self, state):
        ret = (np.sign(np.array(state.head) - np.array(state.food)).tolist() + [state.dir])
        # print ret
        return tuple(ret)

    def initQ(self, input = None):
        if input is None:
            self.Q = Counter()
        else:
            self.Q = Counter(self.load_model(input))

    def updateQ(self, state, action, reward, next_state):
        actions = Action.all()
        Q_ns = [ self.Q[(next_state, a)] for a in range(len(actions)) ]
        # print reward, self.params.discount, Q_ns

        # The crucial Q-value update step, with a const learning rate ( alpha )
        t1 = self.Q[(state, action)]
        t2 = reward + self.params.discount*(np.max(Q_ns))
        self.Q[(state, action)] = (1 - self.params.alpha)*t1 + self.params.alpha*t2

    def learn(self):
        actions = Action.all()
        for i in range(self.params.iter):
            action = Action.random()
            next_state, reward = self.env.submit_action(self.state, action)

            # Logging stats
            print self.state, actions[action], reward, next_state

            # Train the RL agent
            f_state = self.extract_features(self.state)
            f_next_state = self.extract_features(next_state)
            # print f_state, f_next_state
            self.updateQ(f_state, action, reward, f_next_state)

            self.state = next_state

            # time.sleep(0.1)

    def predict(self, state):
        # Predicts the best action to be taken, given the state.
        actions = Action.all()
        Q_s = [ self.Q[(state, a)] for a in range(len(actions)) ]
        action = np.argmax(Q_s)
        return action

    def run(self):
        self.env = Environment(test = True)
        self.state = self.env.init()

        while True:
            f_state = self.extract_features(self.state)
            action = self.predict(f_state)
            next_state, reward = self.env.submit_action(self.state, action)

            if reward == self.env.rewards.win:
                print 'Agent won!'
                break

            # Delaying the simulation
            time.sleep(0.3)

            self.state = next_state

    def dump_model(self, out_dir):
        with open(out_dir + os.sep + 'model.pkl', 'wb') as f:
            pickle.dump(self.Q, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, input_path):
        with open(input_path, 'rb') as f:
            return pickle.load(f)
