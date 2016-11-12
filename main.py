import numpy as np
import os, sys, argparse

import time
import cv2
import cPickle as pickle

from util import *

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

def generate_location(x_range, y_range, arr = None):
    x = np.random.random_integers(x_range[1]) - 1 + x_range[0]
    y = np.random.random_integers(y_range[1]) - 1 + y_range[0]
    pt = (x, y)
    if arr is None:
        return pt
    if pt in arr:
        return generate_location(x_range, y_range, arr)
    else:
        return (x, y)

def clip_position(pt, x_range, y_range):
    x, y = pt
    # x = max(x, x_range[0])
    # x = min(x, x_range[1])
    # y = max(y, y_range[0])
    # y = min(y, y_range[1])
    x = x % ( x_range[1] + 1 )
    y = y % ( y_range[1] + 1 )
    return (x, y)

DIM = 5
MUL_FACTOR = 40

class Params():
    pass

class Environment:
    def __init__(self, test = False):
        # Initialize the environment
        self.width = DIM
        self.height = DIM
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

        self.rewards = Params()
        self.rewards.win = 10
        self.rewards.live = 0

    def init(self):
        self.generate_food()

        # Generate a random (x, y) - snake_head
        snake_head = generate_location(self.x_range, self.y_range, [self.food])
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
        if self.config[next_head] == Type.food:
            self.generate_food([next_head])
        self.config[next_head] = Type.snake

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
        mul = MUL_FACTOR
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

        self.params = Params()
        self.params.alpha = 0.5
        self.params.discount = 0.9

        self.initQ()

    def extract_features(self, state):
        ret = (np.sign(np.array(state.head) - np.array(state.food)).tolist() + [state.dir])
        # print ret
        return tuple(ret)

    def initQ(self):
        self.Q = Counter()

    def updateQ(self, state, action, reward, next_state):
        Q_ns = [ self.Q[(next_state, a)] for a in range(len(actions)) ]
        print reward, self.params.discount, Q_ns
        t1 = self.Q[(state, action)]
        t2 = reward + self.params.discount*(np.max(Q_ns))

        self.Q[(state, action)] = (1 - self.params.alpha)*t1 + self.params.alpha*t2

    def learn(self):
        for i in range(10000):
            action = Actions.random()
            next_state, reward = self.env.submit_action(self.state, action)

            # Logging stats
            print self.state, actions[action], reward, next_state

            # Train the RL agent
            f_state = self.extract_features(self.state)
            f_next_state = self.extract_features(next_state)
            print f_state, f_next_state
            self.updateQ(f_state, action, reward, f_next_state)

            self.env.visualise()
            self.state = next_state

            # time.sleep(0.1)

    def predict(self, state):
        Q_s = [ self.Q[(state, a)] for a in range(len(actions)) ]
        action = np.argmax(Q_s)
        return action

    def run(self):
        self.env = Environment(test = True)
        self.state = self.env.init()

        self.env.visualise()

        while True:
            f_state = self.extract_features(self.state)
            action = self.predict(f_state)
            next_state, reward = self.env.submit_action(self.state, action)

            if reward == self.env.rewards.win:
                print 'Agent won'
                break

            self.env.visualise()
            time.sleep(0.5)

            self.state = next_state

    def dump_model(self, out_dir):
        with open(out_dir + os.sep + 'model.pkl', 'wb') as f:
            pickle.dump(self.Q, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, input_path):
        with open(input_path, 'rb') as f:
            return pickle.load(f)

def main(args):
    agent = Agent()
    if args.learn:
        agent.learn()
        if args.output is not None:
            agent.dump_model(args.output)
    else:
        if args.input is None:
            print 'Input model to be provided if agent should not be learned'
            print 'Aborting..'
            sys.exit()
        else:
            agent.load_model(args.input)

    # Learned agent available

    if args.test:
        again = True
        while again:
            agent.run()
            again = (raw_input('Continue ( Y/N ) ? ').lower() == 'y')

    print 'Done'

if __name__ == '__main__':
    # Parsing command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learn', help='Learn the RL agent', action = 'store_true')
    parser.add_argument('-t', '--test', help='Test the RL agent on random environments', action = 'store_true')
    parser.add_argument('-o', '--output', help='Output directory for storing trained model')
    parser.add_argument('-i', '--input', help='Input file path for the stored model')

    args = parser.parse_args()

    main(args)
