import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

num_workers = 0

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = num_workers,

):
    
    """
    Creates train and test dataloaders.

    Takes in a training and testing directory, and converts them to Pytorch Datasets and
    then Dataloaders.

    Args:
        train_dir (str): Path to training data.
        test_dir (str): Path to testing data.
        transform (transforms.Compose): A composition of Pytorch transforms.
        batch_size (int): Number of images to pass through the model at a time.
        num_workers (int): Number of workers to use for loading data.

    Returns:

        A tuple of train, test dataloaders and class_names.


    """

    # Create train and test datasets
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data, 
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    
    return train_dataloader, test_dataloader, class_names



    