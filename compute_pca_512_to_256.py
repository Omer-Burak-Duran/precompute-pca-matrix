# compute_pca_512_to_256.py
import argparse, json, numpy as np
from sklearn.decomposition import PCA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, required=True,
                    help="Path to coco_train2017_clip_vitb32_embeddings.npy")
    ap.add_argument("--out_prefix", type=str, default="pca_clip_512_to_256")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X = np.load(args.embeddings)  # shape (N, 512), float32
    print("Loaded embeddings:", X.shape, X.dtype)

    pca = PCA(n_components=256, random_state=args.seed)
    pca.fit(X)

    print("Explained variance ratio (first 10):",
          np.round(pca.explained_variance_ratio_[:10], 4))
    print("Total explained variance:", np.round(pca.explained_variance_ratio_.sum(), 4))

    mean = pca.mean_.astype("float32")              # (512,)
    comps = pca.components_.astype("float32")       # (256, 512)

    # Save JSON for Flutter (easiest to load)
    data = {"mean": mean.tolist(), "components": comps.tolist()}
    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump(data, f)

    # Optional: also save as .npy
    np.save("pca_mean.npy", mean)
    np.save("pca_components.npy", comps)

    print("Saved:",
          f"{args.out_prefix}.json, pca_mean.npy, pca_components.npy")

if __name__ == "__main__":
    main()
