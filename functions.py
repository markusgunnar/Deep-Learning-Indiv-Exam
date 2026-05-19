import pandas as pd
from pathlib import Path
import os
import tensorflow as tf

def load_images_from_dir(_path: str, _img_size: tuple):
    temp = tf.keras.utils.image_dataset_from_directory(
        _path,
        labels="inferred",
        label_mode="int",
        image_size=_img_size,
        batch_size=32,
        shuffle=True,
        seed=42
    )

    AUTOTUNE = tf.data.AUTOTUNE
    dataset = temp.prefetch(AUTOTUNE)

    return dataset

def return_class_count_df(_path: str) -> pd.DataFrame:
    data = []
    for class_folder in Path(_path).iterdir():
        if class_folder.is_dir():
            count = len(list(class_folder.glob("*")))
            data.append({
                "Class": class_folder.name,
                "Images": count
            })

    return pd.DataFrame(data)


def count_files(_path: str) -> int:
    count = 0
    for root, dirs, files in os.walk(_path):
        count += len(files)
    return count