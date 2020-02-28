import os

DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "frames")

TMP_DIR = os.path.join(DATA_DIR, "tmp")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")

POS_DIR = os.path.join(DATA_DIR, "detections")
DET_DATA_DIR = os.path.join(DATA_DIR, "detection_data")

FTS_DIR = os.path.join(DATA_DIR, "detections_embeddings")
TRACK_DIR = os.path.join(DATA_DIR, "trajectories")
REF_DIR = os.path.join(DATA_DIR, "ref_trajectories")

