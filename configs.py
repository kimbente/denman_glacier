# HYPERS

dfNN_SIM_RESULTS_DIR = "results/dfNN"
NN_SIM_RESULTS_DIR = "results/NN"
PINN_RESULTS_DIR = "results/PINN"

dfNN_LR = 0.005
NN_LR = 0.005
PINN_LR = 0.005

WEIGHT_DECAY = 0.005

SUBSAMPLE_RATE = 10

# Toggle emission tracking with codecarbon on or off
TRACK_EMISSIONS_BOOL = False
# TRACK_EMISSIONS_BOOL = True

# Define how often to print training progress
PRINT_FREQUENCY = 50

NUM_RUNS = 1 # 8
MAX_NUM_EPOCHS = 2000 # 2000

PATIENCE = 25 # Stop after {PATIENCE} epochs with no improvement

BATCH_SIZE = 1024

# PINN HYPERPARAM (SIM & REAL)
W_PINN_DIV_WEIGHT = 0.3
