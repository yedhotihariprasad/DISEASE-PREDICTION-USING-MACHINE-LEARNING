import os
import argparse
import random
import shutil


def split_dataset(src_dir, dest_dir, split=0.8, seed=42):
    if not os.path.exists(src_dir):
        raise SystemExit(f'source not found: {src_dir}')
    os.makedirs(dest_dir, exist_ok=True)
    train_root = os.path.join(dest_dir, 'train')
    val_root = os.path.join(dest_dir, 'val')
    for root in (train_root, val_root):
        os.makedirs(root, exist_ok=True)

    random.seed(seed)
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    if not classes:
        raise SystemExit('no class subfolders found in source dir')

    for cls in classes:
        src_cls = os.path.join(src_dir, cls)
        files = [f for f in os.listdir(src_cls) if os.path.isfile(os.path.join(src_cls, f))]
        random.shuffle(files)
        split_idx = int(len(files) * split)
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        dest_train_cls = os.path.join(train_root, cls)
        dest_val_cls = os.path.join(val_root, cls)
        os.makedirs(dest_train_cls, exist_ok=True)
        os.makedirs(dest_val_cls, exist_ok=True)

        for fn in train_files:
            shutil.copy2(os.path.join(src_cls, fn), os.path.join(dest_train_cls, fn))
        for fn in val_files:
            shutil.copy2(os.path.join(src_cls, fn), os.path.join(dest_val_cls, fn))

        print(f'class {cls}: {len(train_files)} train, {len(val_files)} val')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='Source dataset dir with class subfolders')
    p.add_argument('--dest', required=True, help='Destination root for train/ and val/')
    p.add_argument('--split', type=float, default=0.8, help='Train fraction (0-1)')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    split_dataset(args.src, args.dest, split=args.split, seed=args.seed)
