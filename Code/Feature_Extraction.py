# Path to images and labels
# -------------------------- Paths for PyCharm --------------------------

SAVE_DIR = "../dataset_as_pkl"
TRAIN_FLOAT32_PATH = "../dataset_as_pkl/normalized_float32_train.pkl"
TEST_FLOAT32_PATH = "../dataset_as_pkl/normalized_float32_test.pkl"
TRAIN_UINT8_PATH = "../dataset_as_pkl/normalized_uint8_train.pkl"
TEST_UINT8_PATH = "../dataset_as_pkl/normalized_uint8_test.pkl"

# -------------------------- Paths for PyCharm --------------------------

# -------------------------- Paths for Colab --------------------------

# DATASET_PATH = "/content/drive/MyDrive/Mask-Or-No-Mask/dataset_as_pkl"
# SAVE_DIR = f"{DATASET_PATH}/../dataset_as_pkl"
# TRAIN_FLOAT32_PATH = f"{DATASET_PATH}/../dataset_as_pkl/normalized_float32_train.pkl"
# TEST_FLOAT32_PATH = f"{DATASET_PATH}/../dataset_as_pkl/normalized_float32_test.pkl"
# TRAIN_UINT8_PATH = f"{DATASET_PATH}/../dataset_as_pkl/normalized_uint8_train.pkl"
# TEST_UINT8_PATH = f"{DATASET_PATH}/../normalized_uint8_test.pkl"

# -------------------------- Paths for Colab --------------------------

# -------------------------- Imports --------------------------
from skimage.feature import hog
from Preprocess import *
import pickle
# -------------------------- Imports --------------------------

# -------------------------- Pickle --------------------------
def save_features(filepath, features, labels):
    with open(filepath, "wb") as f:
        pickle.dump((features, labels), f)

def load_features(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)
# -------------------------- Pickle --------------------------

def get_hog_features(set_type="train", save_path=None):
    if set_type == "train":
        X, y = load_features(TRAIN_FLOAT32_PATH)
    elif set_type == "test":
        X = load_features(TEST_FLOAT32_PATH)
    else:
        raise ValueError('set_type must be train or test')

    hog_features = []

    for image in X:
        # Taken from Lab 5
        HOG_des, _ = hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(8, 8), block_norm="L2-Hys", visualize=True)
        hog_features.append(HOG_des)

    if save_path:
        save_features(save_path, hog_features, y)

    return np.array(hog_features)

# Some code taken from Lab 6
def get_sift_and_orb_features(set_type="train", save_path_sift=None, save_path_orb=None):
    # Load from uint8 .pkl dataset (used for keypoint descriptors)
    if set_type == "train":
        X, y = load_features(TRAIN_UINT8_PATH)
    elif set_type == "test":
        X, y = load_features(TEST_UINT8_PATH)
    else:
        raise ValueError("set_type must be 'train' or 'test'")


    sift = cv2.SIFT_create() # Taken from lab 6
    orb = cv2.ORB_create()

    sift_features, sift_labels = [], []
    orb_features, orb_labels = [], []

    for i, img in enumerate(X):

        kp, sift_desc = sift.detectAndCompute(img, None) # SIFT taken from lab 6
        sift_features.append(sift_desc)
        sift_labels.append(y[i])

        # ORB
        kp, orb_desc = orb.detectAndCompute(img, None)
        orb_features.append(orb_desc)
        orb_labels.append(y[i])

    if save_path_sift:
        save_features(save_path_sift, sift_features, sift_labels)

    if save_path_orb:
        save_features(save_path_orb, orb_features, orb_labels)

    return (sift_features, sift_labels), (orb_features, orb_labels)

if __name__ == "__main__":
    import os

    # ---------------- CONFIGURATION ----------------
    set_type = "train"  # Change to "test" if needed
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Running feature extraction on {set_type} set...")

    # ---------------- HOG ----------------
    hog_save_path = os.path.join(SAVE_DIR, f"hog_{set_type}.pkl")
    print("Extracting HOG features...")
    hog_feats = get_hog_features(set_type=set_type, save_path=hog_save_path)
    print(f"HOG features saved to {hog_save_path}")

    # ---------------- SIFT & ORB ----------------
    sift_save_path = os.path.join(SAVE_DIR, f"sift_{set_type}.pkl")
    orb_save_path = os.path.join(SAVE_DIR, f"orb_{set_type}.pkl")

    print("🔍 Extracting SIFT and ORB features...")
    get_sift_and_orb_features(
        set_type=set_type,
        save_path_sift=sift_save_path,
        save_path_orb=orb_save_path
    )
    print(f"SIFT features saved to {sift_save_path}")
    print(f"ORB features saved to {orb_save_path}")

    print("Feature extraction finished.")

