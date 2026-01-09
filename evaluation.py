import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from PIL import Image

def train_step(model: torch.nn.Module,
               dataset: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = 'cuda'):
    model.eval()
    train_loss, train_acc = 0,0 
    for batch, (X,y) in enumerate(dataset):

        X, y = X.to(device), y.to(device)

        y_logits = model(X)

        loss = loss_fn(y_logits,y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred = torch.argmax(torch.softmax(y_logits,dim=1), dim=1)
        train_acc += (y_pred == y).sum().item()/len(y_logits)
    train_loss = train_loss / len(dataset)
    train_acc = train_acc / len(dataset)
    return train_loss, train_acc



def test_step(model: torch.nn.Module,
              dataset: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device = 'cuda'):
    test_loss, test_acc = 0,0
    for batch, (X,y) in enumerate(dataset):
        with torch.inference_mode():
            X,y = X.to(device), y.to(device)
            
            y_logits = model(X)

            loss = loss_fn(y_logits,y)
            test_loss += loss.item()

            y_pred = torch.argmax(torch.softmax(y_logits,dim=1), dim=1)
            test_acc += (y_pred == y).sum().item()/len(y_logits)
    test_acc = test_acc / len(dataset)
    test_loss = test_loss / len(dataset)
    return test_loss, test_acc




# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataset=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataset=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # 6. Return the filled results at the end of the epochs
    return results


def model_eval(model:torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device = 'cuda'):
    
    pre_loss= 0
    test_pred = []
    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        model.eval()
        with torch.inference_mode():
            
            y_logits = model(X)
            y_pred = torch.argmax(torch.softmax(y_logits, dim=1),dim=1)
            test_pred.append(y_pred)
            loss = loss_fn(y_logits,y)
            pre_loss += loss.item()
    pre_loss /= len(dataloader)
    print(pre_loss)
    return test_pred
