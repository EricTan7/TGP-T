from yacs.config import CfgNode as CN

###########################
# Config definition
###########################

_C = CN()

_C.VERSION = 1

# Directory to save the output files (like log.txt and model weights)
_C.OUTPUT_DIR = "./output"
# Path to a directory where the files were saved previously
_C.RESUME = ""
# Set seed to negative value to randomize everything
# Set seed to positive value to use a fixed seed
_C.SEED = 1
_C.USE_CUDA = True
# Print detailed information
# E.g. trainer, dataset, and backbone
_C.VERBOSE = True
# Use WANDB to log the runs
_C.WANDB = False

###########################
# Input
###########################
_C.INPUT = CN()
_C.INPUT.SIZE = (224, 224)
# Mode of interpolation in resize functions
_C.INPUT.INTERPOLATION = "bilinear"
# For available choices please refer to transforms.py
_C.INPUT.TRANSFORMS = ()
# If True, tfm_train and tfm_test will be None
_C.INPUT.NO_TRANSFORM = False
# Mean and std (default: ImageNet)
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Random crop
_C.INPUT.CROP_PADDING = 4
# Random resized crop
_C.INPUT.RRCROP_SCALE = (0.08, 1.0)
# Cutout
_C.INPUT.CUTOUT_N = 1
_C.INPUT.CUTOUT_LEN = 16
# Gaussian noise
_C.INPUT.GN_MEAN = 0.0
_C.INPUT.GN_STD = 0.15
# RandomAugment
_C.INPUT.RANDAUGMENT_N = 2
_C.INPUT.RANDAUGMENT_M = 10
# ColorJitter (brightness, contrast, saturation, hue)
_C.INPUT.COLORJITTER_B = 0.4
_C.INPUT.COLORJITTER_C = 0.4
_C.INPUT.COLORJITTER_S = 0.4
_C.INPUT.COLORJITTER_H = 0.1
# Random gray scale's probability
_C.INPUT.RGS_P = 0.2
# Gaussian blur
_C.INPUT.GB_P = 0.5  # propability of applying this operation
_C.INPUT.GB_K = 21  # kernel size (should be an odd number)
# Image views
_C.INPUT.IMG_VIEWS = 1
# Text augmentation
_C.INPUT.TEXT_AUG = "hand_crafted"

###########################
# Dataset
###########################
_C.DATASET = CN()
# Directory where datasets are stored
_C.DATASET.ROOT = "/mnt/sdb/tanhao/recognition/"
_C.DATASET.NAME = ""
# List of source/target domains' names (strings)
# Do not apply to some datasets, which have pre-defined splits
_C.DATASET.SOURCE_DOMAINS = ()
_C.DATASET.TARGET_DOMAINS = ()
# Number of labeled instances in total
# Useful for the semi-supervised learning
_C.DATASET.NUM_LABELED = -1
# Number of images per class
_C.DATASET.NUM_SHOTS = -1
# Percentage of validation data (only used for SSL datasets)
# Set to 0 if do not want to use val data
# Using val data for hyperparameter tuning was done in Oliver et al. 2018
_C.DATASET.VAL_PERCENT = 0.1
# Fold index for STL-10 dataset (normal range is 0 - 9)
# Negative number means None
_C.DATASET.STL10_FOLD = -1
# CIFAR-10/100-C's corruption type and intensity level
_C.DATASET.CIFAR_C_TYPE = ""
_C.DATASET.CIFAR_C_LEVEL = 1
# Use all data in the unlabeled data set (e.g. FixMatch)
_C.DATASET.ALL_AS_UNLABELED = False

###########################
# Dataloader
###########################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
# Apply transformations to an image K times (during training)
_C.DATALOADER.K_TRANSFORMS = 1
# img0 denotes image tensor without augmentation
# Useful for consistency learning
_C.DATALOADER.RETURN_IMG0 = False
# Setting for the train_x data-loader
_C.DATALOADER.TRAIN_X = CN()
_C.DATALOADER.TRAIN_X.SAMPLER = "RandomSampler"
_C.DATALOADER.TRAIN_X.BATCH_SIZE = 32
# Parameter for RandomDomainSampler
# 0 or -1 means sampling from all domains
_C.DATALOADER.TRAIN_X.N_DOMAIN = 0
# Parameter of RandomClassSampler
# Number of instances per class
_C.DATALOADER.TRAIN_X.N_INS = 16

# Setting for the train_u data-loader
_C.DATALOADER.TRAIN_U = CN()
# Set to false if you want to have unique
# data loader params for train_u
_C.DATALOADER.TRAIN_U.SAME_AS_X = True
_C.DATALOADER.TRAIN_U.SAMPLER = "RandomSampler"
_C.DATALOADER.TRAIN_U.BATCH_SIZE = 32
_C.DATALOADER.TRAIN_U.N_DOMAIN = 0
_C.DATALOADER.TRAIN_U.N_INS = 16

# Setting for the test data-loader
_C.DATALOADER.TEST = CN()
_C.DATALOADER.TEST.SAMPLER = "SequentialSampler"
_C.DATALOADER.TEST.BATCH_SIZE = 32

###########################
# Model
###########################
_C.MODEL = CN()
# Path to model weights (for initialization)
_C.MODEL.INIT_WEIGHTS = ""
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = ""
_C.MODEL.BACKBONE.PRETRAINED = True
# Definition of embedding layers
_C.MODEL.HEAD = CN()
# If none, do not construct embedding layers, the
# backbone's output will be passed to the classifier
_C.MODEL.HEAD.NAME = ""
# Structure of hidden layers (a list), e.g. [512, 512]
# If undefined, no embedding layer will be constructed
_C.MODEL.HEAD.HIDDEN_LAYERS = ()
_C.MODEL.HEAD.ACTIVATION = "relu"
_C.MODEL.HEAD.BN = True
_C.MODEL.HEAD.DROPOUT = 0.0
# LORA config
_C.MODEL.LORA = CN()
_C.MODEL.LORA.RANK = 1
_C.MODEL.LORA.ALPHA = 0.
_C.MODEL.LORA_V = True
_C.MODEL.LORA_T = True
_C.MODEL.LORA_ATTN = False
# Bonder config
_C.MODEL.BONDER = CN()
_C.MODEL.BONDER.DEPTH = 1
_C.MODEL.BONDER.NUM_Q_CTG = 32
_C.MODEL.BONDER.NUM_Q_CON = 64


###########################
# Optimization
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = "adam"
_C.OPTIM.LR = 0.0003
_C.OPTIM.WEIGHT_DECAY = 1e-4
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.SGD_DAMPNING = 0
_C.OPTIM.SGD_NESTEROV = False
_C.OPTIM.RMSPROP_ALPHA = 0.99
# The following also apply to other
# adaptive optimizers like adamw
_C.OPTIM.ADAM_BETA1 = 0.9
_C.OPTIM.ADAM_BETA2 = 0.999
# STAGED_LR allows different layers to have
# different lr, e.g. pre-trained base layers
# can be assigned a smaller lr than the new
# classification layer
_C.OPTIM.STAGED_LR = False
_C.OPTIM.NEW_LAYERS = ()
_C.OPTIM.BASE_LR_MULT = 0.1
# Learning rate scheduler
_C.OPTIM.LR_SCHEDULER = "single_step"
# -1 or 0 means the stepsize is equal to max_epoch
_C.OPTIM.STEPSIZE = (-1, )
_C.OPTIM.GAMMA = 0.1
_C.OPTIM.MAX_EPOCH = 10
# Set WARMUP_EPOCH larger than 0 to activate warmup training
_C.OPTIM.WARMUP_EPOCH = -1
# Either linear or constant
_C.OPTIM.WARMUP_TYPE = "linear"
# Constant learning rate when type=constant
_C.OPTIM.WARMUP_CONS_LR = 1e-5
# Minimum learning rate when type=linear
_C.OPTIM.WARMUP_MIN_LR = 1e-5
# Recount epoch for the next scheduler (last_epoch=-1)
# Otherwise last_epoch=warmup_epoch
_C.OPTIM.WARMUP_RECOUNT = True
_C.OPTIM.MAX_ITER = 12800
_C.OPTIM.WARMUP_ITER = 50
_C.OPTIM.WARMUP_LR = 1e-5

###########################
# Test
###########################
_C.TEST = CN()
_C.TEST.EVALUATOR = "Classification"
_C.TEST.PER_CLASS_RESULT = False
# Compute confusion matrix, which will be saved
# to $OUTPUT_DIR/cmat.pt
_C.TEST.COMPUTE_CMAT = False
# If NO_TEST=True, no testing will be conducted
_C.TEST.NO_TEST = False
# Use test or val set for FINAL evaluation
_C.TEST.SPLIT = "test"
# Which model to test after training (last_step or best_val)
# If best_val, evaluation is done every epoch (if val data
# is unavailable, test data will be used)
_C.TEST.FINAL_MODEL = "last_step"

###########################
# Train
###########################
_C.TRAIN = CN()
_C.TRAIN.PRINT_FREQ = 10
_C.TRAIN.TEST_FREQ = 5
_C.TRAIN.SAVE_FREQ = 20
_C.TRAIN.DIST_TRAIN = False
_C.TRAIN.SAVE_MODEL = True

###########################
# Trainer
###########################
_C.TRAINER = CN()
_C.TRAINER.NAME = ""
_C.TRAINER.PREC = "fp32"
_C.TRAINER.CTX_INIT = ""

