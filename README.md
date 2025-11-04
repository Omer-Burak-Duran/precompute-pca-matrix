# CLIP PCA Precompute (512 → 256) for AI Photo Finder

This repository contains the scripts and instructions we used to compute a PCA transform that reduces **CLIP ViT-B/32** embeddings from **512-d float32** to **256-d float32**. The resulting PCA matrix is intended to be bundled with our **Flutter AI-assisted photo finder app**, where it will be applied on-device to every image/text embedding produced by our quantized CLIP encoders (ONNX Runtime, *.ort models).

> TL;DR: We precompute PCA offline (on a diverse public dataset), export a JSON with `mean` and `components`, ship that JSON inside the Flutter app, and apply `y = (x - mean) · Wᵀ` on-device to get a compact 256-d embedding. Later, we quantize 256-d float32 to 256-d int8 in the app (quantization is out of scope here).

---

## Why do we need PCA?

Our mobile app indexes and searches **tens of thousands** of on-device photos using semantic embeddings from CLIP (ViT-B/32). Naively storing and searching **512-d float32** vectors can be expensive in **storage, memory, and bandwidth**, especially on mobile devices. PCA helps by:

- **Reducing dimensionality**: 512 → 256 halves the vector size, reducing storage and memory footprint.
- **Speeding up vector operations**: Smaller vectors speed up similarity search and downstream compute (distance calculations, database I/O).
- **Minimal impact on quality**: With a sufficiently diverse PCA training set, 256 components typically retain a large portion of variance and preserve retrieval quality in practice.

We then (optionally) apply **int8 quantization** on the 256-d float32 vectors inside the app to further reduce size/cost; that step is separate from this repo.

---

## Where will we use the PCA?

In our **Flutter** app (Android/iOS), which:
- uses **ONNX Runtime** to run **quantized CLIP ViT-B/32** image and text encoders on-device,
- produces **512-d float32** vectors,
- **L2-normalizes** them (consistent with CLIP usage),
- applies this **precomputed PCA (512→256)** on-device using the shipped `pca_clip_512_to_256.json`,
- (later) quantizes the 256-d float32 to **256-d int8** for storage/search.

---

## Dataset used to fit PCA

We used the **MS COCO 2017 Train** split (~118k images). It’s a widely-used, diverse dataset of everyday scenes (people, objects, indoor/outdoor), and serves as a good approximation of user photo distributions for PCA purposes.

You can optionally add more or different datasets in the future; PCA can be recomputed using the same scripts and workflow.

---

## Repository contents

- `embed_coco_openclip.py`  
  Walks the `train2017/` folder, embeds images with **open_clip** (ViT-B/32), L2-normalizes, and saves a single `.npy` array of shape `(N,512)` plus a `paths.txt` file aligned with the rows.

- `compute_pca_512_to_256.py`  
  Loads the `(N,512)` embeddings, fits a **PCA(n_components=256)** with scikit-learn, and exports:
  - `pca_clip_512_to_256.json` (float32 numbers) with:
    - `"mean"`: length-512 list (feature-wise mean),
    - `"components"`: 256×512 list (each row is one principal component).
  - `pca_mean.npy` and `pca_components.npy` (optional Python-friendly backups).

- `test_apply_pca.py`  
  Minimal check that loads the JSON and applies `y = (x - mean) @ components.T` to a few embeddings, verifying output shape `(?,256)`.

- `requirements.txt`  
  Python dependencies excluding torch/torchvision (install PyTorch with the CUDA index separately).

---

## Setup (Windows 11 friendly)

1) **Create and activate a virtual environment**
```

python -m venv .venv
..venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

```

2) **Install PyTorch (CUDA 13.0 wheels, as used in our environment)**  
> If you don’t have CUDA or want CPU-only, follow the official PyTorch instructions instead.
```

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

```

3) **Install the rest of the dependencies**
```

pip install -r requirements.txt

```

---

## Download COCO 2017 Train

You need the `train2017/` image folder extracted locally. Example (PowerShell):

```

New-Item -ItemType Directory -Path D:\datasets\coco2017 -Force | Out-Null
Set-Location D:\datasets\coco2017

curl -L -o train2017.zip https://images.cocodataset.org/zips/train2017.zip
Expand-Archive -Path .\train2017.zip -DestinationPath .\

# You should now have D:\datasets\coco2017\train2017*.jpg

```

Make sure you have ~40 GB free (zip + extracted), and that `train2017/` contains ~118k images.

---

## Step 1 — Embed images with open_clip (ViT-B/32)

Run Script A to generate embeddings. Adjust paths as needed.

```

python embed_coco_openclip.py --data_dir D:\datasets\coco2017 --batch_size 128 --num_workers 0

```

Outputs:
- `coco_train2017_clip_vitb32_embeddings.npy`  (shape `(N,512)`, float32)
- `coco_train2017_paths.txt`                  (one image path per row)

> Notes:
> - The script uses **open_clip** ViT-B/32 (`laion400m_e32`) and **L2-normalizes** outputs.
> - It’s GPU-accelerated if CUDA is available; otherwise, CPU inference (slower).
> - On Windows, the script writes to a temp memmap and compacts to `.npy` safely.

---

## Step 2 — Compute PCA (512 → 256) and export

Run Script B:

```

python compute_pca_512_to_256.py --embeddings coco_train2017_clip_vitb32_embeddings.npy

````

Outputs:
- `pca_clip_512_to_256.json`  ← **This is the only file you need inside the Flutter app**
- `pca_mean.npy`
- `pca_components.npy`

The JSON contains:
```json
{
  "mean": [512 floats...],
  "components": [[512 floats...], ..., 256 rows total]
}
````

We cast to float32 before saving to keep the file reasonably small.

---

## (Optional) Step 3 — Quick test

```
python test_apply_pca.py
```

You should see something like: `Y shape: (5, 256)`.

---

## Integrating in the Flutter app

1. Place `pca_clip_512_to_256.json` in your project assets, e.g. `assets/models/pca_clip_512_to_256.json`.

2. Add to `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/models/pca_clip_512_to_256.json
```

3. Load and apply on-device (Dart pseudo-code):

```dart
// load JSON once at startup
final jsonStr = await rootBundle.loadString('assets/models/pca_clip_512_to_256.json');
final p = jsonDecode(jsonStr);
final mean = (p['mean'] as List).map((e) => (e as num).toDouble()).toList(); // len 512
final components = (p['components'] as List)
    .map((row) => (row as List).map((e) => (e as num).toDouble()).toList())
    .toList(); // 256 x 512

// given a 512-d CLIP embedding (float32) from ONNX Runtime:
List<double> l2normalize(List<double> v) {
  double s = 0.0; for (final x in v) s += x*x;
  final inv = s > 0 ? 1.0 / s.sqrt() : 1.0;
  return v.map((x) => x * inv).toList();
}

List<double> applyPCA(List<double> x512, List<double> mean, List<List<double>> W) {
  final out = List<double>.filled(W.length, 0.0); // 256
  for (int j = 0; j < W.length; j++) {
    double acc = 0.0;
    final wj = W[j];
    for (int i = 0; i < mean.length; i++) {
      acc += (x512[i] - mean[i]) * wj[i];
    }
    out[j] = acc;
  }
  return out;
}

final x512 = /* CLIP embedding from ONNX */;
final x512n = l2normalize(x512);        // keep consistent with training
final y256 = applyPCA(x512n, mean, components);
```

4. (Later) Quantize `y256` to int8 per your chosen scheme.

---

## Re-running / Recomputing

If you change:

* CLIP model checkpoint,
* normalization,
* or the dataset used for PCA,

you should recompute the PCA to match the new distribution and save a new JSON (version the filename accordingly, e.g., `pca_clip_vitb32_laion400m_e32_512to256.json`).

---

## Troubleshooting

* **Windows file locking (memmap)**
  If you see a `PermissionError: [WinError 32]` when deleting a `.mmp` file, make sure to fully delete references to the memmap and `gc.collect()` before removing the file (the provided script already does this).

* **CUDA vs CPU**
  If CUDA isn’t available or PyTorch CUDA wheels aren’t installed, open_clip will still run on CPU—just slower. Confirm device selection in the script output.

* **RAM usage**
  `N × 512 × 4 bytes ~ 2 KB per vector`. For ~118k images, embeddings are ~242 MB. With headroom, this should fit in desktop memory. If you scale much larger, consider `IncrementalPCA`.

---

## Credits

* CLIP model by OpenAI; open-source implementations via **open_clip**.
* PCA via **scikit-learn**.
* COCO dataset authors and maintainers.
