import os
import cv2
import numpy as np
from ultralytics import YOLO, FastSAM
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, filedialog
import requests
import sys

# Default config
DEFAULT_CONFIG = {
    "BASE_DIR": "/main/folder/to/datasets/",
    "SPLITS": ["train", "valid", "test"],
    "OUTPUT_BASE_DIR": "folde/to/save/segmented/dataset/",
    "FASTSAM_CHECKPOINT": "FastSAM-x.pt",
    "YOLO_MODEL": "path/to/your/yolo/detect/model/",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "BATCH_SIZE": 10,
    "NUM_WORKERS": 4,
    "CONF_THRESHOLD": 0.1
}

FASTSAM_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-x.pt"

def download_fastsam_checkpoint(path):
    if not os.path.exists(path):
        print(f"FastSAM checkpoint not found at {path}. Downloading from {FASTSAM_URL}...")
        try:
            response = requests.get(FASTSAM_URL, stream=True)
            response.raise_for_status()
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded FastSAM checkpoint to {path}")
        except Exception as e:
            print(f"Failed to download FastSAM checkpoint: {e}")
            sys.exit(1)
    else:
        print(f"FastSAM checkpoint found at {path}")

class ConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentation Config")
        self.config = DEFAULT_CONFIG.copy()
        self.base_dir_var = tk.StringVar(value=self.config["BASE_DIR"])
        self.output_dir_var = tk.StringVar(value=self.config["OUTPUT_BASE_DIR"])
        self.fastsam_checkpoint_var = tk.StringVar(value=self.config["FASTSAM_CHECKPOINT"])
        self.yolo_model_var = tk.StringVar(value=self.config["YOLO_MODEL"])
        self.device_var = tk.StringVar(value=self.config["DEVICE"])
        self.batch_size_var = tk.IntVar(value=self.config["BATCH_SIZE"])
        self.num_workers_var = tk.IntVar(value=self.config["NUM_WORKERS"])
        self.conf_threshold_var = tk.DoubleVar(value=self.config["CONF_THRESHOLD"])
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.root, text="Base Directory:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(self.root, textvariable=self.base_dir_var, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.root, text="Browse", command=lambda: self.browse_dir(self.base_dir_var)).grid(row=0, column=2, padx=5)

        ttk.Label(self.root, text="Output Directory:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(self.root, textvariable=self.output_dir_var, width=40).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.root, text="Browse", command=lambda: self.browse_dir(self.output_dir_var)).grid(row=1, column=2, padx=5)

        ttk.Label(self.root, text="FastSAM Checkpoint Path:").grid(row=2, column=0, padx=5, pady=5)
        ttk.Entry(self.root, textvariable=self.fastsam_checkpoint_var, width=40).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(self.root, text="Browse", command=lambda: self.browse_file(self.fastsam_checkpoint_var, "*.pt")).grid(row=2, column=2, padx=5)

        ttk.Label(self.root, text="YOLO Model Path:").grid(row=3, column=0, padx=5, pady=5)
        ttk.Entry(self.root, textvariable=self.yolo_model_var, width=40).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(self.root, text="Browse", command=lambda: self.browse_file(self.yolo_model_var, "*.pt")).grid(row=3, column=2, padx=5)

        ttk.Label(self.root, text="Device (cuda/cpu):").grid(row=4, column=0, padx=5, pady=5)
        ttk.Combobox(self.root, textvariable=self.device_var, values=["cuda", "cpu"], state="readonly").grid(row=4, column=1, padx=5, pady=5)

        ttk.Label(self.root, text="Batch Size:").grid(row=5, column=0, padx=5, pady=5)
        ttk.Entry(self.root, textvariable=self.batch_size_var, width=10).grid(row=5, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(self.root, text="Number of Workers:").grid(row=6, column=0, padx=5, pady=5)
        ttk.Entry(self.root, textvariable=self.num_workers_var, width=10).grid(row=6, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(self.root, text="Confidence Threshold:").grid(row=7, column=0, padx=5, pady=5)
        ttk.Entry(self.root, textvariable=self.conf_threshold_var, width=10).grid(row=7, column=1, padx=5, pady=5, sticky="w")

        ttk.Button(self.root, text="Run Segmentation", command=self.run).grid(row=8, column=0, columnspan=3, pady=10)

    def browse_dir(self, var):
        dir_path = filedialog.askdirectory()
        if dir_path:
            var.set(dir_path)

    def browse_file(self, var, file_type):
        file_path = filedialog.askopenfilename(filetypes=[("Model files", file_type)])
        if file_path:
            var.set(file_path)

    def run(self):
        self.config["BASE_DIR"] = self.base_dir_var.get()
        self.config["OUTPUT_BASE_DIR"] = self.output_dir_var.get()
        self.config["FASTSAM_CHECKPOINT"] = self.fastsam_checkpoint_var.get()
        self.config["YOLO_MODEL"] = self.yolo_model_var.get()
        self.config["DEVICE"] = self.device_var.get()
        self.config["BATCH_SIZE"] = self.batch_size_var.get()
        self.config["NUM_WORKERS"] = self.num_workers_var.get()
        self.config["CONF_THRESHOLD"] = self.conf_threshold_var.get()
        self.root.destroy()

def load_models(config):
    yolo = YOLO(config["YOLO_MODEL"], task='detect')
    yolo.fuse()  # Call once to handle fusing
    yolo.to(config["DEVICE"])
    download_fastsam_checkpoint(config["FASTSAM_CHECKPOINT"])
    fastsam = FastSAM(config["FASTSAM_CHECKPOINT"])
    fastsam.to(config["DEVICE"])
    return yolo, fastsam

def detect_bboxes(image_path, yolo, conf_threshold):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    height, width = img.shape[:2]
    
    results = yolo.predict(image_path, conf=conf_threshold, verbose=True)  # Verbose for debugging
    bboxes = []
    all_boxes = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            x_min, y_min, x_max, y_max = map(float, box.xyxy[0])
            w_abs = x_max - x_min
            h_abs = y_max - y_min
            all_boxes.append(f"Class={class_id}, Conf={confidence:.2f}, Box=[{x_min:.2f}, {y_min:.2f}, {x_max:.2f}, {y_max:.2f}]")
            if class_id != 0:  # Assuming 'player' is class 0; adjust if needed
                continue
            if w_abs < 5 or h_abs < 5 or x_min < 0 or y_min < 0 or x_max > width or y_max > height:
                print(f"Skipping invalid box in {image_path}: x_min={x_min}, y_min={y_min}, w={w_abs}, h={h_abs}, img_size={width}x{height}")
                continue
            bboxes.append([x_min, y_min, x_max, y_max])
    
    if all_boxes:
        print(f"All detections in {image_path}: {all_boxes}")
    if not bboxes:
        print(f"No valid 'player' detections after filtering in {image_path}")
    return img, bboxes

def mask_to_yolo_polygons(mask, img_shape):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or len(contours[0]) < 3:
        return []
    contour = contours[0].flatten()
    height, width = img_shape
    normalized = [coord / width if i % 2 == 0 else coord / height for i, coord in enumerate(contour)]
    return normalized

def process_image(image_info, yolo, fastsam, config):
    split, img_name = image_info
    img_path = os.path.join(config["BASE_DIR"], split, "images", img_name)
    output_mask_dir = os.path.join(config["OUTPUT_BASE_DIR"], split, "masks")
    output_img_path = os.path.join(config["OUTPUT_BASE_DIR"], split, "images", img_name)
    output_label_path = os.path.join(config["OUTPUT_BASE_DIR"], split, "labels", img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

    result = detect_bboxes(img_path, yolo, config["CONF_THRESHOLD"])
    if result is None:
        print(f"Skipping {img_name} in {split}: Image invalid")
        return None
    image, bboxes = result
    if not bboxes:
        print(f"Skipping {img_name} in {split}: No detections")
        return None

    height, width = image.shape[:2]
    print(f"Processing {img_name} in {split}: Image size={width}x{height}, Boxes={bboxes}")

    masks = []
    for box in bboxes:
        results = fastsam.predict(img_path, bboxes=[box], conf=0.4, iou=0.9, retina_masks=True, device=config["DEVICE"])
        if results and len(results) > 0 and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            masks.append(mask)
        else:
            print(f"No mask generated by FastSAM for box {box} in {img_name}")
            # Fallback: Use GrabCut with safer rectangle handling
            mask = np.zeros((height, width), np.uint8)
            x_min, y_min, x_max, y_max = [int(b) for b in box]
            w = x_max - x_min
            h = y_max - y_min
            # Expand rectangle slightly to ensure bg/fg samples
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)
            w = x_max - x_min
            h = y_max - y_min
            if w < 10 or h < 10:  # Skip if still too small
                print(f"Skipping GrabCut for {img_name}: Rectangle too small after padding (w={w}, h={h})")
                continue
            rect = (x_min, y_min, w, h)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
                masks.append(mask)
            except cv2.error as e:
                print(f"GrabCut failed for {img_name} with box {box}: {e}")
                continue

    if not masks:
        print(f"Skipping {img_name} in {split}: No masks generated")
        return None

    output_masks = []
    polygons = []
    for mask in masks:
        polygon = mask_to_yolo_polygons(mask, (height, width))
        if polygon:
            polygons.append(polygon)
        else:
            print(f"No valid polygon generated for a mask in {img_name}")

    with open(output_label_path, "w") as f:
        for i, (mask, polygon) in enumerate(zip(masks, polygons)):
            mask_path = os.path.join(output_mask_dir, f"{img_name.split('.')[0]}_{i}.png")
            cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
            output_masks.append(mask_path)
            f.write(f"0 {' '.join(map(str, polygon))}\n")

    cv2.imwrite(output_img_path, image)
    return output_masks

if __name__ == "__main__":
    root = tk.Tk()
    app = ConfigGUI(root)
    root.mainloop()

    config = app.config

    for split in config["SPLITS"]:
        os.makedirs(os.path.join(config["OUTPUT_BASE_DIR"], split, "images"), exist_ok=True)
        os.makedirs(os.path.join(config["OUTPUT_BASE_DIR"], split, "labels"), exist_ok=True)
        os.makedirs(os.path.join(config["OUTPUT_BASE_DIR"], split, "masks"), exist_ok=True)

    yolo, fastsam = load_models(config)

    all_images = []
    for split in config["SPLITS"]:
        img_dir = os.path.join(config["BASE_DIR"], split, "images")
        if os.path.exists(img_dir):
            for img_name in os.listdir(img_dir):
                if img_name.endswith((".jpg", ".png", ".jpeg")):
                    all_images.append((split, img_name))

    batches = [all_images[i:i + config["BATCH_SIZE"]] for i in range(0, len(all_images), config["BATCH_SIZE"])]
    print(f"Processing {len(all_images)} images across {len(batches)} batches...")
    with ThreadPoolExecutor(max_workers=config["NUM_WORKERS"]) as executor:
        for _ in tqdm(executor.map(lambda b: [process_image(img, yolo, fastsam, config) for img in b], batches), total=len(batches)):
            pass

    print(f"Finished! Segmented dataset saved in {config['OUTPUT_BASE_DIR']}")