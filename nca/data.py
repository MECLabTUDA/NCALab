# TODO: download, verify checksum, unzip
# TODO: dataset class for images / class labels



class SegmentationDataset(Dataset):
    def __init__(self, image_path, label_path, num_channels_total: int):
        """
        :param num_channels_total: Total number of channels our NCA model processes.
        """
        super(self, SegmentationDataset).__init__()
        self.num_channels_total = num_channels_total

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class ClassificationDataset(Dataset):
    def __init__(self, image_path, classes: dict, num_channels_total: int):
        """
        :param classes: Dict mapping image filenames to respective class IDs
        :param num_channels_total: Total number of channels our NCA model processes.
        """
        super(self, SegmentationDataset).__init__()
        self.num_channels_total = num_channels_total

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass