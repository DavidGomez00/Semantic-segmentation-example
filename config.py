# Training hiperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 1
NUM_EPOCHS = 1
MIN_EPOCHS = 1

# Dataset
DATA_DIR = "data/"
NUM_WORKERS = 7
IMAGE_HEIGHT=160
IMAGE_WIDTH=240

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = "bf16-mixed"
