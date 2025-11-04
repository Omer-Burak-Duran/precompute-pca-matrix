# embed_coco_openclip_v1_1.py
import os, sys, json, math, time, argparse
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as td
from PIL import Image, UnidentifiedImageError
import open_clip
from tqdm import tqdm

def list_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for dp, _, files in os.walk(root):
        for f in files:
            if Path(f).suffix.lower() in exts:
                paths.append(os.path.join(dp, f))
    paths.sort()
    return paths

class ImgDataset(td.Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
            img = self.preprocess(img)
            return img, idx
        except (UnidentifiedImageError, OSError):
            return None, idx

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Path containing train2017/")
    ap.add_argument("--subset", type=int, default=0, help="Use only first N images (0=all)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0, help="Windows-safe default=0")
    ap.add_argument("--model", type=str, default="ViT-B-32-quickgelu")
    ap.add_argument("--pretrained", type=str, default="laion400m_e32")
    ap.add_argument("--out_prefix", type=str, default="coco_train2017")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train2017"
    assert train_dir.is_dir(), f"train2017 folder not found at {train_dir}"

    paths = list_images(str(train_dir))
    if args.subset and args.subset > 0:
        paths = paths[:args.subset]
    n = len(paths)
    print(f"Found {n} images.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()

    ds = ImgDataset(paths, preprocess)
    dl = td.DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=(device=="cuda"), drop_last=False)

    # temp memmap file for sequential writes
    temp_mmp = f"{args.out_prefix}_clip_vitb32_embeddings.mmp"
    feats = np.memmap(temp_mmp, dtype="float32", mode="w+", shape=(n, 512))
    valid_mask = np.zeros(n, dtype=np.uint8)

    for batch, idxs in tqdm(dl, total=math.ceil(n/args.batch_size)):
        keep = [i for i, x in enumerate(batch) if x is not None]
        if len(keep) == 0:
            continue
        batch = torch.stack([batch[i] for i in keep]).to(device)
        idxs = [idxs[i].item() for i in keep]

        img_feats = model.encode_image(batch)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        img_feats = img_feats.float().cpu().numpy()  # (B,512)

        for j, row_idx in enumerate(idxs):
            feats[row_idx, :] = img_feats[j]
            valid_mask[row_idx] = 1

    # ---- Windows-safe compaction ----
    valid_idx = np.where(valid_mask == 1)[0]
    feats.flush()

    # Materialize subset to RAM to release mapping
    all_feats = np.array(feats[valid_idx], dtype=np.float32, copy=True)

    # Fully close/unmap before deletion on Windows
    del feats
    import gc
    gc.collect()

    os.remove(temp_mmp)

    # Save outputs
    out_feats = f"{args.out_prefix}_clip_vitb32_embeddings.npy"
    out_paths = f"{args.out_prefix}_paths.txt"
    np.save(out_feats, all_feats)
    with open(out_paths, "w", encoding="utf-8") as f:
        for i in valid_idx:
            f.write(paths[i] + "\n")

    print("Done.")
    print("Embeddings:", out_feats, all_feats.shape)
    print("Paths:", out_paths)

if __name__ == "__main__":
    main()
