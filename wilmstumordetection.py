

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple

# Data science / vision imports
try:
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from albumentations import (
        HorizontalFlip, VerticalFlip, RandomRotate90, Transpose, ShiftScaleRotate,
        Blur, OpticalDistortion, GridDistortion, ElasticTransform, CLAHE, RandomBrightnessContrast,
        Compose, BboxParams
    )
    from ultralytics import YOLO
except Exception as e:
    # If imports fail, provide a clear error; users should install dependencies in their environment.
    raise ImportError(f"Missing dependency: {e}. Install required packages (see requirements.txt).")


def augmentation_pipeline():
    """Return an albumentations Compose pipeline with bbox support (YOLO format)."""
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=45, p=0.5),
        Blur(blur_limit=3, p=0.2),
        OpticalDistortion(p=0.3),
        GridDistortion(p=0.3),
        ElasticTransform(p=0.1),
        CLAHE(p=0.5),
        RandomBrightnessContrast(p=0.5)
    ], bbox_params=BboxParams(format='yolo', label_fields=['category_id']))


def read_yolo_labels(label_path: Path) -> List[List[str]]:
    """Read a YOLO-format label file and return a list of token lists per line."""
    labels = []
    if not label_path.exists():
        return labels
    with label_path.open('r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                labels.append(parts)
    return labels


def write_yolo_labels(label_path: Path, labels: List[List[float]]):
    """Write YOLO-format labels to a file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open('w') as f:
        for label in labels:
            f.write(' '.join(map(str, label)) + "\n")


def augment_data(input_image_dir: Path, input_label_dir: Path, output_image_dir: Path, output_label_dir: Path,
                 aug_per_image: int = 16):
    """Augment images and corresponding YOLO labels and save them to output directories."""
    pipe = augmentation_pipeline()
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in os.listdir(input_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    count = 0
    for image_file in image_files:
        image_path = input_image_dir / image_file
        label_path = input_label_dir / (image_path.stem + '.txt')

        if not label_path.exists():
            # Skip images without labels (optionally log this).
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: failed to read image {image_path}; skipping.")
            continue

        labels = read_yolo_labels(label_path)
        if not labels:
            continue

        bboxes = []
        categories = []
        for parts in labels:
            # Expect: class x_center y_center width height
            if len(parts) >= 5:
                cls = int(float(parts[0]))
                x_center, y_center, width, height = map(float, parts[1:5])
                bboxes.append([x_center, y_center, width, height])
                categories.append(cls)

        for i in range(aug_per_image):
            augmented = pipe(image=image, bboxes=bboxes, category_id=categories)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']

            aug_labels = []
            for bbox, cat in zip(aug_bboxes, categories):
                x_center, y_center, width, height = bbox
                aug_labels.append([cat, x_center, y_center, width, height])

            aug_image_file = f"aug_{image_path.stem}_{i}.jpg"
            aug_image_path = output_image_dir / aug_image_file
            aug_label_path = output_label_dir / (aug_image_path.stem + '.txt')

            cv2.imwrite(str(aug_image_path), aug_image)
            write_yolo_labels(aug_label_path, aug_labels)
            count += 1

    print(f"Generated {count} augmented images and annotations.")


def split_dataset(augmented_image_dir: Path, augmented_label_dir: Path, output_dataset_dir: Path, split_ratio: float = 0.8, seed: int = 42):
    """Split augmented images into train/val and copy to a YOLOv8-compatible folder structure."""
    images_train_dir = output_dataset_dir / 'images' / 'train'
    images_val_dir = output_dataset_dir / 'images' / 'val'
    labels_train_dir = output_dataset_dir / 'labels' / 'train'
    labels_val_dir = output_dataset_dir / 'labels' / 'val'

    images_train_dir.mkdir(parents=True, exist_ok=True)
    images_val_dir.mkdir(parents=True, exist_ok=True)
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)

    augmented_images = [f for f in os.listdir(augmented_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.Random(seed).shuffle(augmented_images)
    train_size = int(len(augmented_images) * split_ratio)

    train_images = augmented_images[:train_size]
    val_images = augmented_images[train_size:]

    def copy_files(image_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
        for image_file in image_list:
            label_file = os.path.splitext(image_file)[0] + '.txt'
            shutil.copy(src_img_dir / image_file, dst_img_dir / image_file)
            shutil.copy(src_lbl_dir / label_file, dst_lbl_dir / label_file)

    copy_files(train_images, augmented_image_dir, augmented_label_dir, images_train_dir, labels_train_dir)
    copy_files(val_images, augmented_image_dir, augmented_label_dir, images_val_dir, labels_val_dir)

    print(f"Copied {len(train_images)} training images and {len(val_images)} validation images.")


def create_dataset_yaml(output_dataset_dir: Path, nc: int = 2, names: List[str] = None):
    """Create dataset.yaml for YOLOv8 training."""
    if names is None:
        names = ["other_tumor", "wilms_tumor"]
    dataset_yaml = f"""train: {output_dataset_dir / 'images' / 'train'}

val: {output_dataset_dir / 'images' / 'val'}

nc: {nc}
names: {names}
\"\"\"
    yaml_path = output_dataset_dir / 'dataset.yaml'
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open('w') as f:
        f.write(dataset_yaml)
    print(f"Created dataset yaml at {yaml_path}")


def train_yolo(data_yaml: Path, model_name: str = 'yolov8n.pt', epochs: int = 80, imgsz: int = 320, batch: int = 16, device: str = None, half: bool = True, patience: int = 300):
    """Train YOLOv8 model using ultralytics.YOLO API. Returns the model object."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = YOLO(model_name)
    results = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=device, half=half, patience=patience)
    return model, results


def copy_experiment_outputs(default_output_dir: Path, destination_base: Path):
    """Copy YOLOv8 experiment output directories to a destination on completion."""
    if not default_output_dir.exists():
        print(f"Default output directory {default_output_dir} does not exist; nothing to copy.")
        return
    experiment_dirs = [d for d in os.listdir(default_output_dir) if os.path.isdir(default_output_dir / d)]
    for exp_dir in experiment_dirs:
        src = default_output_dir / exp_dir
        dst = destination_base / exp_dir
        if dst.exists():
            print(f"Destination {dst} already exists; skipping copy for {exp_dir}.")
            continue
        shutil.copytree(src, dst)
        print(f"Copied experiment {exp_dir} to {dst}")


def evaluate_and_plot(results_csv_path: Path, save_graph_dir: Path):
    """Load training results CSV and plot metrics.
    Expects results CSV created by YOLO training (results.csv)."""
    if not results_csv_path.exists():
        print(f"Results CSV {results_csv_path} not found. Skipping plots.")
        return

    save_graph_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.read_csv(results_csv_path)
    results_df.columns = results_df.columns.str.strip()

    # compute train/val composite losses if columns exist
    if all(col in results_df.columns for col in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']):
        results_df['train_loss'] = (results_df['train/box_loss'] + results_df['train/cls_loss'] + results_df['train/dfl_loss']) / 3
    if all(col in results_df.columns for col in ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']):
        results_df['val_loss'] = (results_df['val/box_loss'] + results_df['val/cls_loss'] + results_df['val/dfl_loss']) / 3
    if all(col in results_df.columns for col in ['metrics/precision(B)', 'metrics/recall(B)']):
        results_df['f1_score'] = 2 * (results_df['metrics/precision(B)'] * results_df['metrics/recall(B)']) / (results_df['metrics/precision(B)'] + results_df['metrics/recall(B)'])

    # simple plotting helper
    def plot_metric(x, y, name, fname):
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label=name)
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.title(f'{name} over Epochs')
        plt.grid(True)
        plt.legend()
        plt.savefig(save_graph_dir / fname)
        plt.close()

    # Plot if columns exist
    if 'train_loss' in results_df.columns:
        plot_metric(results_df['epoch'], results_df['train_loss'], 'train_loss', 'train_loss.png')
    if 'val_loss' in results_df.columns:
        plot_metric(results_df['epoch'], results_df['val_loss'], 'val_loss', 'val_loss.png')
    if 'metrics/mAP50(B)' in results_df.columns:
        plot_metric(results_df['epoch'], results_df['metrics/mAP50(B)'], 'mAP_50', 'mAP50.png')

    print(f"Saved plots to {save_graph_dir}")


def run_inference(model: YOLO, image_dir: Path, device: str = None) -> List[int]:
    """Run inference on all images in image_dir using ultralytics YOLO model and return predicted class labels (first detection per image or -1)."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    preds = []
    image_paths = sorted([image_dir / f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            preds.append(-1)
            continue
        # Run model directly with ultralytics API (it handles preprocessing)
        results = model(str(image_path))
        if len(results) == 0 or len(results[0].boxes) == 0:
            preds.append(-1)
        else:
            # take the first detected class
            cls = int(results[0].boxes.cls[0].item())
            preds.append(cls)
    return preds


def dice_coefficient(y_true, y_pred, smooth=1):
    import torch as _torch
    y_true_f = _torch.tensor(y_true).float().view(-1)
    y_pred_f = _torch.tensor(y_pred).float().view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def parse_args():
    parser = argparse.ArgumentParser(description='Wilms Tumor Detection - cleaned script')
    parser.add_argument('--input_images', type=str, default='data/images', help='Path to input images')
    parser.add_argument('--input_labels', type=str, default='data/labels', help='Path to input labels')
    parser.add_argument('--aug_images', type=str, default='data/augmented/images', help='Path to save augmented images')
    parser.add_argument('--aug_labels', type=str, default='data/augmented/labels', help='Path to save augmented labels')
    parser.add_argument('--yolo_dataset', type=str, default='data/yolov8dataset', help='Path to YOLOv8 dataset output')
    parser.add_argument('--results_csv', type=str, default='runs/results.csv', help='Path to results.csv produced by YOLO training')
    parser.add_argument('--save_graph_dir', type=str, default='runs/graphs', help='Where to save metric graphs')
    parser.add_argument('--train', action='store_true', help='If set, run training (may be slow)')
    parser.add_argument('--augment', action='store_true', help='If set, run augmentation step')
    parser.add_argument('--epochs', type=int, default=80, help='Training epochs')
    parser.add_argument('--model_name', type=str, default='yolov8n.pt', help='Base YOLOv8 weights')
    return parser.parse_args()


def main():
    args = parse_args()

    input_image_dir = Path(args.input_images)
    input_label_dir = Path(args.input_labels)
    aug_image_dir = Path(args.aug_images)
    aug_label_dir = Path(args.aug_labels)
    yolo_dataset_dir = Path(args.yolo_dataset)
    results_csv_path = Path(args.results_csv)
    save_graph_dir = Path(args.save_graph_dir)

    # Basic safety checks
    if args.augment:
        if not input_image_dir.exists() or not input_label_dir.exists():
            raise FileNotFoundError("Input images or labels directory not found. Set correct paths or disable --augment.")
        augment_data(input_image_dir, input_label_dir, aug_image_dir, aug_label_dir, aug_per_image=16)
        split_dataset(aug_image_dir, aug_label_dir, yolo_dataset_dir)
        create_dataset_yaml(yolo_dataset_dir, nc=2, names=['other_tumor', 'wilms_tumor'])

    if args.train:
        yaml_path = yolo_dataset_dir / 'dataset.yaml'
        if not yaml_path.exists():
            raise FileNotFoundError(f"Dataset YAML not found at {yaml_path}. Run augmentation to create it.")
        model, results = train_yolo(yaml_path, model_name=args.model_name, epochs=args.epochs)

        # Copy outputs to a safe location if needed
        copy_experiment_outputs(Path('runs/detect'), Path('yolov8_complete_output'))

    # Plot evaluation metrics if results CSV exists
    evaluate_and_plot(results_csv_path, save_graph_dir)

    print("Script finished. Edit and run with appropriate flags (--augment, --train).")


if __name__ == '__main__':
    main()
