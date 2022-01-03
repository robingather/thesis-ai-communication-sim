from collections import namedtuple
import torch as T
import numpy as np

PopPair = namedtuple('PopPair','pred, prey')

# WORLD
WORLD_SIZE = (192,128)
N_OBST = int(250)
N_PREY = int(12)
N_PRED = int(8)

# MODEL
MODEL_NAME = 'Wa-hearcom'
LOAD_MODEL = True
SAVE_MODEL = False #!
N_INPUTS = PopPair(14, 14)
STATE_LABELS = ['P-lr','P-ud','D-l','D-r','D-u','D-d','Cw-l','Cw-r','Cw-u','Cw-d','Cb-l','Cb-r','Cb-u','Cb-d']
ACTION_LABELS = ['L','R','U','D','COM','X']
N_ACTIONS = 6
N_HIDDEN = 36
SECOND_HIDDEN = False
ABSOLUTE_COM = False
LEARN = PopPair(True, True)

# GENETIC ALGO
POP_AMOUNT =  PopPair(64,128)
MUTATION_CHANCE = 0.01
SUCCESSION_AMOUNT = 8
FITNESS_FUNCTION = PopPair('eaten','survived') # killed, eaten, or survived
CROSSOVER_TYPE = 'split' # uniform or split
GEN_AMOUNT = -1 # amount of PRED generations to run for (-1 means no limit)
DEVICE = 'cpu' #T.device('cuda:0' if T.cuda.is_available() else 'cpu')

# AGENTS
RAD_COM = 64 # radius of communication range in blocks
MAX_HEALTH = PopPair(200,300)
COMMUNICATE_WITHIN_POP = PopPair(True, False) # can speak and listen to each other
HEAR_BETWEEN_POP = PopPair(False, True) # can hear other group
PREY_MOVE = True
OBSTACLE_SIGHT_RANGE = 8

# GRAPHICS
BLOCK_SIZE = 5
SHOW_WORLD = True
RENDER_COMMUNICATION = True
SPEED = 100_000 # user input handles this.
PLOT_GRAPHS = False
PLOT_LAST_GEN = True