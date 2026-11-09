
from pathlib import Path
import torch

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.parent / "Model" / "data"

TRAIN_FILE = DATA_DIR / "train.npz"
VAL_FILE = DATA_DIR / "val.npz"
TEST_FILE = DATA_DIR / "test.npz"

CLIP_FEATURE_FILE = DATA_DIR / "clip_features_mind3d.npz"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

# ============================================================
# DATA DIMENSIONS
# ============================================================

FMRI_FRAMES = 10
FMRI_FEATURES = 31141

NUM_VIEWS = 32
IMAGE_SIZE = 64
CLIP_DIM = 512

NUM_POINTS = 2048
POINT_DIM = 3

# ============================================================
# NEURO-FUSION ENCODER
# ============================================================

NFE_HIDDEN_DIM = 512
NFE_OUTPUT_DIM = 512
NFE_TEMPORAL_LAYERS = 2
NFE_DROPOUT = 0.1

# ============================================================
# FEATURE BRIDGE DIFFUSION
# ============================================================

FBDM_HIDDEN_DIM = 256
FBDM_NUM_LAYERS = 4
FBDM_NUM_HEADS = 8
FBDM_TIMESTEPS = 100
FBDM_DROPOUT = 0.1

# ============================================================
# LATENT ADAPTED DECODER
# ============================================================

LAD_CONDITION_DIM = 512
LAD_HIDDEN_DIM = 256
LAD_NUM_LAYERS = 4
LAD_DROPOUT = 0.1

# ============================================================
# TRAINING
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE_NFE = 16
BATCH_SIZE_FBDM = 16
BATCH_SIZE_LAD = 8

NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()

NFE_EPOCHS = 100
FBDM_EPOCHS = 100
LAD_EPOCHS = 150

NFE_LEARNING_RATE = 1e-4
FBDM_LEARNING_RATE = 1e-4
LAD_LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 15

# ============================================================
# LOSSES
# ============================================================

CONTRASTIVE_TEMPERATURE = 0.07

LAMBDA_CHAMFER = 1.0
LAMBDA_REPULSION = 0.01

# ============================================================
# DIFFUSION
# ============================================================

BETA_START = 1e-4
BETA_END = 0.02

# ============================================================
# INFERENCE
# ============================================================

INFERENCE_BATCH_SIZE = 4
GENERATED_POINTS = NUM_POINTS

# ============================================================
# GPU / REPRODUCIBILITY
# ============================================================

USE_AMP = torch.cuda.is_available()


# ============================================================
# COMPATIBILITY ALIASES FOR TRAINING / INFERENCE SCRIPTS
# ============================================================

NORMALIZATION_PATH = CHECKPOINT_DIR / "fmri_normalization.npz"

DROPOUT = 0.1

NFE_BATCH_SIZE = BATCH_SIZE_NFE
FBDM_BATCH_SIZE = BATCH_SIZE_FBDM
LAD_BATCH_SIZE = BATCH_SIZE_LAD

NFE_LR = NFE_LEARNING_RATE
FBDM_LR = FBDM_LEARNING_RATE
LAD_LR = LAD_LEARNING_RATE

GRAD_CLIP = GRADIENT_CLIP

DIFFUSION_BETA_START = BETA_START
DIFFUSION_BETA_END = BETA_END

CHAMFER_WEIGHT = LAMBDA_CHAMFER
REPULSION_WEIGHT = LAMBDA_REPULSION
