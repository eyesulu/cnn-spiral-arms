import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import time
import copy
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator

def load_dataset(dataset_path, mean, std, batch_size, train_size, test_size):
    """
    Loads and splits an image dataset into training, validation, and test sets with specified transforms.

    Args:
        dataset_path (str): Path to the root directory containing images organized in subfolders by class.
        mean (list of float): Mean values for normalization per channel.
        std (list of float): Standard deviation values for normalization per channel.
        batch_size (int): Number of samples per batch.
        train_size (float): Proportion of the dataset to use for training.
        test_size (float): Proportion of the dataset to use for testing.

    Returns:
        dict: Dataloaders for 'train', 'val', and 'test' splits.
    """

    train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=mean, std=std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)

    total_size = len(dataset)
    train_size = int(train_size * total_size)  
    test_size = int(test_size * total_size)  
    val_size = total_size - train_size - test_size 

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    test_dataset.dataset.transform = test_transform

    dataloaders = {
    'train': DataLoader(train_dataset, batch_size=64, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=64, shuffle=False),
    'test': DataLoader(test_dataset, batch_size=64, shuffle=False)
    }

    return dataloaders

def get_model(model_name, learning_rate, weighting):
    """
    Initializes a model (ResNet101 or VGG16) with pretrained weights, sets up optimizer, scheduler, and loss function.

    Args:
        model_name (str): Either 'ResNet101' or 'VGG16'.
        learning_rate (float): Learning rate for the optimizer.
        weighting (list of float): Class weighting for the loss function.

    Returns:
        tuple: (model, optimizer, scheduler, loss_function, device)
    """
    if model_name == "ResNet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = True
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
    elif model_name == "VGG16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = True
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 2)
    else:
        print("Incorrect model name. Valid choices: 'Resnet101'; 'VGG16'.")
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    decay_learning_rate = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    loss_function = nn.CrossEntropyLoss(weight = torch.tensor(weighting))

    return model, optimizer, decay_learning_rate, loss_function, device

def train_model(model, 
                optimizer, 
                scheduler, 
                loss_function,
                device, 
                dataloaders, 
                num_epochs, 
                patience = 1):
    """
    Trains the model using training and validation data, with early stopping based on validation accuracy.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        loss_function (callable): Loss function.
        device (torch.device): Device to train on.
        dataloaders (dict): Dictionary of dataloaders for 'train' and 'val'.
        dataset_sizes (list): Sizes of the training and validation sets.
        num_epochs (int): Number of epochs to train.
        patience (int, optional): Number of epochs with no improvement before early stopping.

    Returns:
        torch.nn.Module: The trained model.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    consecutive_epochs_without_improvement = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        phases = ['train', 'val']
        for phase in phases:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            dataloader = dataloaders[phase]
            with tqdm(total=len(dataloader), desc=f"{phase} phase", leave=False) as pbar:
                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_function(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    pbar.update(1)

            if phase == 'train':
                scheduler.step()

            if phase == 'val':
                epoch_loss = running_loss / len(dataloaders["val"].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders["val"].dataset)
            elif phase == 'train':
                epoch_loss = running_loss /  len(dataloaders["train"].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders["train"].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    consecutive_epochs_without_improvement = 0
                else:
                    consecutive_epochs_without_improvement += 1

        if consecutive_epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break
    return model

def test_model(model, dataloaders, device, loss_function=None):
    """
    Evaluates the trained model on the test set.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataloaders (dict): Dictionary containing the 'test' dataloader.
        device (torch.device): Device to perform testing on.
        loss_function (callable, optional): Loss function for reporting test loss.

    Returns:
        dict: Dictionary containing test accuracy, optional loss, predictions, and true labels.
    """
    model.eval()
    dataloader = dataloaders["test"]
    
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="test phase", leave=False) as pbar:
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                if loss_function:
                    loss = loss_function(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                pbar.update(1)

    total_samples = len(dataloader.dataset)
    accuracy = running_corrects.double() / total_samples
    avg_loss = running_loss / total_samples if loss_function else None

    print(f"Test Accuracy: {accuracy:.4f}")
    if avg_loss is not None:
        print(f"Test Loss: {avg_loss:.4f}")

    return {
        'accuracy': accuracy.item(),
        'loss': avg_loss,
        'predictions': all_preds,
        'labels': all_labels
    }

def get_filename(prefix, ext, directory="."):
    """
    Generates a filename with a timestamp to prevent overwriting.

    Args:
        prefix (str): Prefix for the filename.
        ext (str): File extension (e.g., 'pth').
        directory (str): Directory to place the file in.

    Returns:
        str: Full path to the uniquely named file.
    """
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{prefix}_{now}.{ext}"
    return os.path.join(directory, filename)

def save_model(model, model_name, version):
    """
    Saves the trained model with a timestamped filename to avoid overwrites.

    Args:
        model (torch.nn.Module): The model to save.
        model_name (str): Either 'ResNet101' or 'VGG16'.
        version (int): Version number for organizing saved models.

    Returns:
        None
    """
    savepath = f"./{model_name}/v{version}"
    os.makedirs(savepath, exist_ok=True)
    final_savepath = get_filename("best_model", "pth", directory=savepath)
    torch.save(model, final_savepath)
    return

def load_latest_model(model_name, version, device):
    """
    Loads the latest model checkpoint from the specified version directory.

    Args:
        model_name (str): Either 'ResNet101' or 'VGG16'.
        version (int): Version number to locate the correct model folder.
        device (torch.device): Device to map the model onto.

    Returns:
        torch.nn.Module: The loaded model.
    """
    models_dir = f"./{model_name}/v{version}"
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError(f"No .pth model files found in {models_dir}")
    
    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)))
    
    latest_model_path = os.path.join(models_dir, model_files[-1])
    print(f"Loading latest model: {latest_model_path}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(latest_model_path, map_location=device)
    return model

def divide_into_grids(image, grid_dims=(5, 5)):
    rows, cols = grid_dims
    w, h = image.size
    grid_w, grid_h = w // cols, h // rows
    return [image.crop((col*grid_w, row*grid_h, (col+1)*grid_w, (row+1)*grid_h))
            for row in range(rows) for col in range(cols)], (grid_w, grid_h)

def get_prediction(model, image_path, n_cols, mean, std):
    """
    Loads an image, splits it into a grid, applies a model to each cell, and visualizes the predictions.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        image_path (str): Path to the image to analyze.
        n_cols (int): Number of columns/rows in the plot grid.
        mean (list of float): Mean values for normalization per channel.
        std (list of float): Standard deviation values for normalization per channel.

    Returns:
        List of confidences for predicted class 1 in the image grid cells.
    """
    data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    ])
    
    model.eval()
    img = Image.open(image_path).convert('RGB')

    grids, grid_dims = divide_into_grids(img, (n_cols, n_cols))
    confidences = []
    results = []

    for grid in grids:
        grid_tensor = data_transform(grid).unsqueeze(0)
        with torch.no_grad():
            outputs = model(grid_tensor)
            probabilities = nn.Softmax(dim=1)(outputs)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
            results.append((predicted_class, confidence))

    fig, axes = plt.subplots(n_cols, n_cols, figsize=(20, 20))

    for idx, (grid, (predicted_class, confidence)) in enumerate(zip(grids, results)):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.imshow(grid)
        ax.axis("off")

        if predicted_class == 1:
            confidences.append(confidence)
            if confidence > 90:
                color = 'darkgreen'
            elif confidence > 70:
                color = 'green'
            else:
                color = 'orange'
            rect = patches.Rectangle((0, 0), grid.size[0], grid.size[1],
                                     linewidth=5, facecolor=color, alpha=0.5)
            ax.add_patch(rect)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.tight_layout(pad=2)
    plt.show()
    plt.close()

    return confidences

def plot_histogram(confidences, fontsize = 19):
    """
    Plots a histogram of confidence levels for regions predicted as class 1.

    Args:
        confidences (list of float): A list of confidence scores (percent values between 0–100)
                                     corresponding to predictions labeled as class 1.

    Returns:
        None. Displays a matplotlib histogram showing the distribution of confidence levels.
    """

    font = {'fontname':'Times New Roman'}
    plt.rcParams.update({'font.size': 19})
    plt.figure(figsize=(10, 5))
    plt.hist(confidences, bins = 15,range=[50, 100], color = 'black', histtype='barstacked', rwidth=0.8 )
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Confidence level (%)', **font)
    plt.ylabel ('Count', **font)
    plt.xticks(**font)
    plt.yticks(**font)
    plt.xticks(range(50, 101, 10))
    plt.tight_layout()
    plt.show()
    plt.close()

