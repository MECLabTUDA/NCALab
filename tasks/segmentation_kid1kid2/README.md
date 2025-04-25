# KID 2 Bleeding Segmentation

This example operates on the proprietary KID 2 dataset for capsule endoscopic bleeding segmentation.
See https://mdss.uth.gr/datasets/endoscopy/kid/ to learn how you get access to the data.


## Convert Data

Before we can use the data, they need to be converted to the nnUNet format.
This is necessary to compare to nnUNet as a baseline with reproducible 5-fold cross-validation splits.

But first of all, you'll need to create a local config file for this task by copying and modifying config.py.example:

```bash
cp config.py.example config.py
```

Open it with your favorite text editor (e.g. vim) and change the KID_DATASET_PATH (directory of KID2 dataset) and KID_DATASET_PATH_NNUNET (where you store your nnUNet datasets).
Once that's done, running the following script will create a new dataset with ID 11 in your nnUNet dataset directory.
You can change the id by passing the `--id` parameter to this script.

```bash
python3 create_nnunet_dataset.py
```

Switch to your nnUNet dataset directory, and follow the official instructions (TODO: Link) to run the baseline training.
First, you'll need to set up your environment variables, `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results`.
Afterwards, run the following lines to preprocess your data and initialize the split:


```bash
nnUNetv2_plan_and_preprocess -d 11 --verify_dataset_integrity
for i in {0..4}; do nnUNetv2_train 11 2d $i; done
```

(You may replace "11" with the dataset ID you assigned earlier.)
It is not necessary to run the full training to initialize the dataset -- we are mainly interested in `nnUNet_preprocessed/Dataset011_KID2vascular/splits_final.json`, which contains a definition of the 5-fold cross-validation split.
However, you'll need to run the full training if you are interested in comparing and reproducing results of course.
