# Mango Leaf Disease Detection (Mango Leaf)

This project provides a minimal end-to-end scaffold to train a mango-leaf disease classifier and serve predictions via a web interface.

Contents:
- `backend/` — Flask API and model loader
- `frontend/` — simple upload UI
- `train/` — training script using TensorFlow/Keras
- `dataset/` — expected dataset layout (not included)
- `tests/` — basic test scaffold

Quick start

1. Install dependencies:

```bash
pip install -r "d:/Desktop/pcl leaf/requirements.txt"
```

2. Prepare dataset:

Place images in `dataset/train/<class_name>/` and `dataset/val/<class_name>/`.

3. Train:

You can point training to any dataset root that contains `train/` and `val/` subfolders. Example using your dataset at `D:\Games\vsco\dataset leaf`:

```bash
python "d:/Desktop/pcl leaf/train/train.py" --data-dir "D:/Games/vsco/dataset leaf"
```

Or specify train/val explicitly:

```bash
python "d:/Desktop/pcl leaf/train/train.py" --train-dir "D:/Games/vsco/dataset leaf/train" --val-dir "D:/Games/vsco/dataset leaf/val"
```

4. Run backend:

```bash
python "d:/Desktop/pcl leaf/backend/app.py"
```

5. Open `frontend/index.html` in a browser and upload an image.
