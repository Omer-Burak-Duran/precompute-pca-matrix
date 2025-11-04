# test_apply_pca.py
import json, numpy as np

# Load outputs from previous steps
X = np.load("coco_train2017_clip_vitb32_embeddings.npy")  # (N,512)
with open("pca_clip_512_to_256.json", "r") as f:
    p = json.load(f)

mean = np.array(p["mean"], dtype=np.float32)              # (512,)
components = np.array(p["components"], dtype=np.float32)  # (256,512)

# Transform first 5 vectors: y = (x - mean) @ components.T
Y = (X[:5] - mean) @ components.T                         # (5,256)
print("Y shape:", Y.shape, Y.dtype)
