import numpy as np
import os, sys, argparse

import time
import cv2
import cPickle as pickle

from util import *
from agent import *

def main(args):
    if args.learn:
        agent = Agent()
        # Agent learns from close interaction with the environment
        agent.learn()
        if args.output is not None:
            agent.dump_model(args.output)
    else:
        if args.input is None:
            print 'Input model to be provided if agent should not be learned'
            print 'Aborting..'
            sys.exit()
        else:
            agent = Agent(args.input)

    # Learned agent available

    if args.test:
        print 'Press any key to run agent on another environment, <Esc> to abort.'
        while True:
            # Runs the agent on a randomly generated environment
            agent.run()
            k = cv2.waitKey(0)
            if (k & 0xFF) == 27:
                break

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
