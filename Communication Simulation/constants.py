from collections import namedtuple

PopPair = namedtuple('PopPair','pred, prey')

# WORLD
WORLD_SIZE = (192,128)
N_OBST = int(250)
N_PREY = int(12)
N_PRED = int(8)

# MODEL
MODEL_NAME = 'Wa-com'
LOAD_MODEL = True
SAVE_MODEL = False #!
N_INPUTS = PopPair(14, 14) # neural network inputs of predators and prey
N_ACTIONS = 6 # amount of actions for agents
N_HIDDEN = 36 # amount of hidden nodes in network
SECOND_HIDDEN = False # enable/disable second hidden layer

# GENETIC ALGO
POP_AMOUNT =  PopPair(64,128) # population sizes for predators and prey
MUTATION_CHANCE = 0.01 # default 1%
SUCCESSION_AMOUNT = 8 # amount of best-performing agents to carry over to next generation (elitism)
FITNESS_FUNCTION = PopPair('eaten','survived') # selection criterions of genetic algorithm: killed, eaten, or survived
CROSSOVER_TYPE = 'split' # uniform or split

# AGENTS
RAD_COM = 64 # radius of communication range in cells
MAX_HEALTH = PopPair(200,300) # maximum health points of predators and prey
LEARN = PopPair(False, False) # enable/disable learning for predators and pey
COMMUNICATE_WITHIN_POP = PopPair(False, False) # enable/disable within-populaiton communication of predators and prey
HEAR_BETWEEN_POP = PopPair(False, False) # enable/disable between-populaiton communication of predators and prey. 1. can predators hear prey, 2. can prey hear predators
PREY_MOVE = True # allow prey to move. 'False' to force all prey to stand still always
ABSOLUTE_COM = False # force communication to be positive

# GRAPHICS
BLOCK_SIZE = 5 # size of cells to render on screen
SHOW_WORLD = True # enable/disable rendering of world
RENDER_COMMUNICATION = True # enable/disable rendering of communication ranges
PLOT_GRAPHS = False # enable/disable plotting of graphs on every generation
PLOT_LAST_GEN = True # enable/disable gathering of last generation statistics and plotting of those statistics. Should be 'False' when learning, 'True' when evaluating.
STATE_LABELS = ['P-lr','P-ud','D-l','D-r','D-u','D-d','Cw-l','Cw-r','Cw-u','Cw-d','Cb-l','Cb-r','Cb-u','Cb-d']
ACTION_LABELS = ['L','R','U','D','COM','X']

# SIMULATION LOOP
DEVICE = 'cpu' #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # run on graphics card or cpu
GEN_AMOUNT = -1 # amount of predator generations to run for (-1 means no limit) (models get saved at the end of run)
SPEED = 100_000 # speed of simulation