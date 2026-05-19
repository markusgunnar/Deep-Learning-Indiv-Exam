import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import os
import tensorflow as tf


def load_images_from_dir(_path: str, _img_size: tuple, _subset: str):
    """
    Loads and coverts images to normalized grayscale
    
    _path:
        Path to data
    _img_size:
        Size of the image as tuple
    _subset:
        Type of dataset. training or validation. empty for test data
    """

    kwargs = {
        "labels": "inferred",
        "label_mode":"int",
        "image_size":_img_size,
        "color_mode":"grayscale",
        "batch_size":32,
        "shuffle":True,
        "seed":42
    }

    if _subset:
        kwargs["validation_split"] = 0.2
        kwargs["subset"] = _subset

    temp = tf.keras.utils.image_dataset_from_directory(
        _path,
        **kwargs
    )

    class_names = temp.class_names

    # Normalize pixel values
    temp = temp.map(
        lambda x, y: (x / 255.0, y)
    )

    AUTOTUNE = tf.data.AUTOTUNE
    dataset = temp.prefetch(AUTOTUNE)

    return dataset, class_names


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


def show_n_images(_img_dataset, _class_names: list, _nr: int):
    plt.figure(figsize=(12,5))

    for images, labels in _img_dataset.take(1):
        for i in range(_nr):
            plt.subplot(2,6,i+1)

            plt.imshow(images[i].numpy().squeeze(), cmap="gray")
            plt.title(_class_names[labels[i]])
            plt.axis("off")

    plt.tight_layout()
    plt.show()