import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import img_to_array
from PIL import Image

from PIL import Image

def load_image_data(base_dir, target_size=(128, 128)):
    X = []
    y = []
    classes = sorted(os.listdir(base_dir))  # Menentukan kelas
    label_map = {class_name: idx for idx, class_name in enumerate(classes)}

    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            img_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            if len(img_files) == 15:  # Pastikan setiap kelas memiliki tepat 15 gambar
                for img_name in img_files:
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        img = Image.open(img_path)
                        if img.mode in ('RGBA', 'P') and 'transparency' in img.info:
                            img = img.convert('RGBA')
                        else:
                            img = img.convert('RGB')

                        img = img.resize(target_size)
                        img_array = img_to_array(img) / 255.0

                        # Pastikan dimensi gambar yang valid (128, 128, 3)
                        if img_array.shape == (target_size[0], target_size[1], 3):
                            X.append(img_array)
                            class_label = label_map[class_name]
                            y.append(class_label)
                        else:
                            print(f"Skipping image {img_name} with invalid shape {img_array.shape}")
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    if np.max(y) >= len(classes):
        print(f"Warning: Found labels outside the valid range. Max label: {np.max(y)}")

    return train_test_split(X, y, test_size=0.2, random_state=42)

