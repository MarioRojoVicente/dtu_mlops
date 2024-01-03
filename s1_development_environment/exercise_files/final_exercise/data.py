import torch


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    train = torch.load("../../../data/corruptmnist/train_images_0.pt")
    train_labels = torch.load("../../../data/corruptmnist/train_target_0.pt")
    test = torch.load("../../../data/corruptmnist/test_images.pt")
    test_labels = torch.load("../../../data/corruptmnist/test_target.pt")
    return lambda: zip(train,train_labels), lambda: zip(test, test_labels)
