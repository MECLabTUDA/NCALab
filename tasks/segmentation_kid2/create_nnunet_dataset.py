import json
from pathlib import Path
import shutil

import pandas as pd
import click
import cv2

from config import KID_DATASET_PATH, KID_DATASET_PATH_NNUNET

TASK_PATH = Path(__file__).parent


@click.command()
@click.option("--id", "-i", default=11, help="nnUNet dataset identifier", type=int)
def main(id):
    # This split definition was originally created for a train / val / test split.
    # For generating a 5-fold cross-validation setup, we concatenate training and validation
    # and generate nnUNet folders accordingly.
    kid2_split_path = Path(TASK_PATH / "split_vascular.csv")
    df = pd.read_csv(kid2_split_path)
    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]

    training_filenames = sorted(pd.concat((df_train["filename"], df_val["filename"])))

    dest_dir = KID_DATASET_PATH_NNUNET / "nnUNet_raw" / f"Dataset{id:03d}_KID2vascular"
    dest_dir.mkdir(exist_ok=True, parents=True)

    with open(dest_dir / "dataset.json", "w") as f:
        json.dump(
            {
                "channel_names": {
                    "0": "rgb_to_0_1",
                    "1": "rgb_to_0_1",
                    "2": "rgb_to_0_1",
                },
                "labels": {"background": 0, "vascular": 1},
                "numTraining": len(training_filenames),
                "file_ending": ".png",
            },
            f,
        )

    # Create nnUNet data directories for training and test sets
    (dest_dir / "imagesTr").mkdir(exist_ok=True)
    (dest_dir / "imagesTs").mkdir(exist_ok=True)
    (dest_dir / "labelsTr").mkdir(exist_ok=True)
    (dest_dir / "labelsTs").mkdir(exist_ok=True)

    # generate training dataset
    for i, training_image_filename in enumerate(training_filenames):
        training_image_path = KID_DATASET_PATH / "vascular" / training_image_filename
        training_label_path = (
            KID_DATASET_PATH
            / "vascular-annotations"
            / (training_image_filename[:-4] + "m.png")
        )
        nnunet_filename = f"KID2V_{i:03d}_0000.png"
        nnunet_label_filename = f"KID2V_{i:03d}.png"
        shutil.copy2(training_image_path, dest_dir / "imagesTr" / nnunet_filename)
        image = cv2.imread(str(training_label_path))
        image = image // 255
        cv2.imwrite(str(dest_dir / "labelsTr" / nnunet_label_filename), image)

    # generate test dataset
    for i, test_image_filename in enumerate(sorted(df_test["filename"])):
        test_image_path = KID_DATASET_PATH / "vascular" / test_image_filename
        test_label_path = (
            KID_DATASET_PATH
            / "vascular-annotations"
            / (test_image_filename[:-4] + "m.png")
        )
        nnunet_filename = f"KID2V_{i:03d}_0000.png"
        nnunet_label_filename = f"KID2V_{i:03d}.png"
        shutil.copy2(test_image_path, dest_dir / "imagesTs" / nnunet_filename)
        image = cv2.imread(str(test_label_path))
        image = image // 255
        cv2.imwrite(str(dest_dir / "labelsTs" / nnunet_label_filename), image)


if __name__ == "__main__":
    main()
