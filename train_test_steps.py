import torch

from tqdm.auto import tqdm
from typing import Tuple, Dict, List

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               ) -> Tuple[float, float]:
    
    """
    Performs a step of training the model on the dataloader 
    
    Args:
        model: torch.nn.Module
            A model to be trained on the dataloader.
        
        dataloader: torch.utils.data.DataLoader
            A dataloader to train the model on.

        loss_fn: torch.nn.Module
            A loss function to minimize.

        optimizer: torch.optim.Optimizer
            An optimizer to help minimize the loss function.

        device: torch.device
            A device to train the model on. For ex: ("cuda" or "cpu")

    Returns:

        A tuple of loss and accuracy.

    

    """



    model.train()

    loss, acc  = 0.0, 0.0

    for X,y in tqdm(dataloader):

        X = X.to(device)
        y = y.to(device)

        yhat = model(X)

        loss = loss_fn(yhat,y)

        loss += loss.item()

        ypred =  torch.argmax(torch.softmax(yhat,dim=1), dim=1)

        acc += (ypred == y).sum().item()/len(y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    loss = loss/len(dataloader)
    acc = acc/len(dataloader)

    return loss, acc


def test_step(model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              device: torch.device) -> Tuple[float, float]:
    
    """
    Performs a step of testing the model on the dataloader
    
    Args:
        model: torch.nn.Module
            A model to be tested on the dataloader.
        
        dataloader: torch.utils.data.DataLoader
            A dataloader to test the model on.

        loss_fn: torch.nn.Module
            A loss function to calculate the test loss.

        device: torch.device
            A device to test the model on. For ex: ("cuda" or "cpu")

    Returns:
            
            A tuple of loss and accuracy.
    
        
    
        
    """

    model.eval()

    loss, acc = 0.0, 0.0

    with torch.inference_mode():

        for X,y in dataloader:

            X = X.to(device)
            y = y.to(device)

            yhat = model(X)

            loss = loss_fn(yhat,y)

            loss += loss.item()

            ypred = torch.argmax(torch.softmax(yhat,dim=1),dim=1)

            acc += (ypred == y).sum().item()/len(y)

    loss = loss/len(dataloader)
    acc = acc/len(dataloader)

    return loss, acc


def train_model(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epochs:int) -> Dict[str,List[float]]:
    
    """
    Trains the model on the train_dataloader and tests it on the test_dataloader for the specified number of epochs
    and returns the training and test losses and accuracies.

    Args:
        model: torch.nn.Module
            A model to be trained on the dataloader.

        train_dataloader: torch.utils.data.DataLoader
            The training dataloader
        
        test_dataloader: torch.utils.data.DataLoader
            The testing dataloader

        loss_fn: torch.nn.Module
            A loss function to minimize.

        optimizer: torch.optim.Optimizer
            An optimizer to help minimize the loss function.

        device: torch.device
            A device to train the model on. For ex: ("cuda" or "cpu")

        epochs: int
            The number of epochs to train the model for.

    """

    results = {"train_loss":[],
               "train_acc":[],
               "test_loss":[],
               "test_acc":[]}

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)

        test_loss, test_acc = test_step(model, loss_fn, test_dataloader, device)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        print(f"Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    return results
    
    

