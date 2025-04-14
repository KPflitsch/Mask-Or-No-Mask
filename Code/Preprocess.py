import os
import pickle
import numpy as np
import cv2

# Path to images and labels
# -------------------------- Paths for PyCharm --------------------------

DATASET_PATH = "../CW_Dataset"
TRAIN_IMAGE_PATH = "../CW_Dataset/train/images"
TRAIN_LABEL_PATH = "../CW_Dataset/train/labels"
TEST_IMAGE_PATH = "../CW_Dataset/test/images"
TEST_LABEL_PATH = "../CW_Dataset/test/labels"
SAVE_DIR = "../dataset_as_pkl"

# -------------------------- Paths for PyCharm --------------------------

# -------------------------- Paths for Colab --------------------------

# DATASET_PATH = "/content/drive/MyDrive/Mask-Or-No-Mask/CW_Dataset"
# TRAIN_IMAGE_PATH = f"{DATASET_PATH}/train/images"
# TRAIN_LABEL_PATH = f"{DATASET_PATH}/train/labels"
# TEST_IMAGE_PATH = f"{DATASET_PATH}/test/images"
# TEST_LABEL_PATH = f"{DATASET_PATH}/test/labels"
# SAVE_DIR = f"{DATASET_PATH}/../normalized_dataset"

# -------------------------- Paths for Colab --------------------------
def load_dataset(image_dir, label_dir, valid_ext=".jpeg"):
    # Create array to store images and labels
    image_label_pairs = []

    # Iterate through all images in dataset
    for img_file in os.listdir(image_dir):
        # Skip invalid images
        if not img_file.lower().endswith(valid_ext):
            continue
        # Point to images
        img_path = os.path.join(image_dir, img_file)

        # Point to corresponding label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)

        # Skip if no label
        if not os.path.exists(label_path):
            continue

        # Read label
        with open(label_path, 'r') as f:
            label_string = f.read().strip()

        # Skip of invalid label
        if label_string not in ['0', '1', '2']:
            continue

        # Convert label (string) to int
        label = int(label_string)
        # Save image and label as tuple
        image_label_pairs.append((img_path, label))

    return image_label_pairs

def resize_with_padding(img, target_size=(64, 64), pad_colour=0):
    # Get image height and width
    h, w = img.shape[:2]
    # Target height and width
    th, tw = target_size

    # Set interpolation function based downscaling or upscaling
    interpolation = cv2.INTER_AREA if h > th or w > tw else cv2.INTER_CUBIC

    # Calculate aspect ratio of input image and maintain it for maintaining that aspect ratio
    aspect_ratio = float(h) / float(w)
    if aspect_ratio > 1:
        new_w = tw
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = th
        new_w = int(new_h * aspect_ratio)

    # Resize the image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    # Calculate amount of padding needed, and keeping image centered
    pad_top = (th - new_h) // 2
    pad_bottom = th - new_h - pad_top
    pad_left = (tw - new_w) // 2
    pad_right = tw - new_w - pad_left

    # Add the padding (chosen colour is black)
    padded_image = cv2.copyMakeBorder(resized_img, pad_top, pad_bottom, pad_left, pad_right, borderType = cv2.BORDER_CONSTANT, value = pad_colour)

    # For SIFT and ORB especially, still works with others
    padded_image = np.clip(padded_image, 0, 255).astype(np.uint8)

    return padded_image

def resize_and_normalize_uint8(image_label_pairs, target_size=(64, 64)):
    X, y = [], []
    for img_path, label in image_label_pairs:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        padded = resize_with_padding(img, target_size)
        X.append(padded)  # uint8
        y.append(label)
    return X, y

def resize_and_normalize_float32(image_label_pairs, target_size=(64, 64)):
    X, y = [], []
    for img_path, label in image_label_pairs:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        padded = resize_with_padding(img, target_size)
        normalized = padded.astype(np.float32) / 255.0
        X.append(normalized)
        y.append(label)
    return X, y

def save_as_pkl(X, y, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump((X, y), f)

if __name__ == "__main__":
    print("Starting preprocessing...")

    # Load image/label pairs
    train_pairs = load_dataset(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH)
    test_pairs = load_dataset(TEST_IMAGE_PATH, TEST_LABEL_PATH)

    print(f"Loaded {len(train_pairs)} training and {len(test_pairs)} test samples")

    # --- For classical ML (uint8) ---
    X_train_uint8, y_train_uint8 = resize_and_normalize_uint8(train_pairs)
    X_test_uint8, y_test_uint8 = resize_and_normalize_uint8(test_pairs)

    save_as_pkl(X_train_uint8, y_train_uint8, os.path.join(SAVE_DIR, "normalized_uint8_train.pkl"))
    save_as_pkl(X_test_uint8, y_test_uint8, os.path.join(SAVE_DIR, "normalized_uint8_test.pkl"))

    print("Saved uint8 preprocessed datasets")

    # --- For deep learning (float32) ---
    X_train_f32, y_train_f32 = resize_and_normalize_float32(train_pairs)
    X_test_f32, y_test_f32 = resize_and_normalize_float32(test_pairs)

    save_as_pkl(X_train_f32, y_train_f32, os.path.join(SAVE_DIR, "normalized_float32_train.pkl"))
    save_as_pkl(X_test_f32, y_test_f32, os.path.join(SAVE_DIR, "normalized_float32_test.pkl"))

    print("All datasets preprocessed and saved to .pkl format")






