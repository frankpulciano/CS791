# Import necessary packages
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as TF
from torch import nn
from torch.utils.data import DataLoader
import wandb
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)

def train(dataloader, resnet_model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    resnet_model.train()
    step = 0
    for batch, (X, y) in enumerate(dataloader):
        wandb.log({"Training_Step": step})
        step +=1
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = resnet_model(X)
        loss = loss_fn(pred, y)

        wandb.log({'Training_Loss-Step': loss})
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, resnet_model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    resnet_model.eval()
    test_loss, distance_from_actual = 0, 0
    step = 0
    with torch.no_grad():
        for X, y in dataloader:
            wandb.log({"Testing_Step": step})
            step += 1
            X, y = X.to(device), y.to(device)
            pred = resnet_model(X)
            loss = loss_fn(pred, y).item()
            test_loss += loss
            wandb.log({'Testing_Loss-Step': loss})
            for index, value in enumerate(y):
                wandb.log({'Distance_From_Actual-Step': abs(pred[index] - value)})
                distance_from_actual += abs(pred[index] - value)
    test_loss /= num_batches
    distance_from_actual /= size
    wandb.log({'Testing_Loss-Epoch': test_loss})
    wandb.log({'Distance_From_Actual-Epoch': distance_from_actual})
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n Avg Distance from actual: {distance_from_actual}")
    return test_loss

class FractureDataset(Dataset):
    def __init__(self, path_to_csv, train, transform=None):
        self.data = pd.read_csv(path_to_csv, sep=',', skiprows=0)
        if train:
            self.data = self.data.head(int(len(self.data)*0.7))
        else:
            self.data = self.data.tail(int(len(self.data)*0.3))
        # Get the paths to each of the input images
        self.input_filenames = self.data["input"]

        # Get the label for each image
        self.img_labels = self.data["label"]

        # If there are any transform functions to be called, store them
        self.transform = transform

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        load_path =  self.input_filenames.iloc[idx]
        image = torchvision.io.read_image(load_path)

        image = TF.convert_image_dtype(image, torch.float)

        # Read in the attribute labels for the current input image
        crack_distance_label = self.img_labels.iloc[idx]
        crack_distance_label = torch.tensor(crack_distance_label).float().squeeze()

        if self.transform:
            image = self.transform(image)
            return image, crack_distance_label
        else:
            return image, crack_distance_label

hyperparameter_defaults = dict(
    epochs=100,
    learning_rate=0.001,
    optimizer="Adam",
    scheduler="ReduceLROnPlateau",
    loss_function="MSELoss",
    image_size="fracture_dataset.csv",
)

wandb.init(config=hyperparameter_defaults, project="fracture", entity="unr-mpl")
config = wandb.config


if config["image_size"] == "fracture_dataset.csv":
    batch_size = 16
elif config["image_size"] == "fracture_dataset_224x168.csv":
    batch_size = 128
elif config["image_size"] == "fracture_dataset_160x120.csv":
    batch_size = 256
training_data = FractureDataset(
    path_to_csv=f"./{config['image_size']}",
    train=True,
    # transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p=0.5),
    #                                           torchvision.transforms.RandomVerticalFlip(p=0.5),
    #                                           ])
)
test_data = FractureDataset(
    path_to_csv=f"./{config['image_size']}",
    train=False,
)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=10, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=10, shuffle=False)
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

 # Define model
class ResNet_classifier(nn.Module):

    # constructor
    def __init__(self):
        super().__init__()
        # self.model = resnet50
        self.model = resnet18
        last_layer_inputs = self.model.fc.in_features
        self.model.fc = nn.Linear(last_layer_inputs, 1, bias=True)
        self.model.to(device)

    def forward(self, x):
        x = self.model.forward(x)
        x = x.squeeze()
        return x


resnet_model = ResNet_classifier()

if config["loss_function"] == "MSELoss":
    loss_fn = nn.MSELoss()
if config["loss_function"] == "L1Loss":
    loss_fn = nn.L1Loss()
if config["loss_function"] == "SmoothL1Loss":
    loss_fn = nn.SmoothL1Loss()
if config["loss_function"] == "HuberLoss":
    loss_fn = nn.HuberLoss()
# optimizer = torch.optim.SGD(resnet_model.parameters(), lr=0.001)
if config["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(resnet_model.parameters(), lr=config["learning_rate"])
elif config["optimizer"] == "AdamW":
    optimizer = torch.optim.AdamW(resnet_model.parameters(), lr=config["learning_rate"])
elif config["optimizer"] == "SGD":
    optimizer = torch.optim.SGD(resnet_model.parameters(), lr=config["learning_rate"])
elif config["optimizer"] == "AdaGrad":
    optimizer = torch.optim.Adagrad(resnet_model.parameters(), lr=config["learning_rate"])
if config["scheduler"] == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
elif config["scheduler"] == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
elif config["scheduler"] == "MultiplicativeLR":
    lmbda = lambda epoch: 0.95
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
elif config["scheduler"] == "None":
    scheduler = None
model = ResNet_classifier()

load = False
if load == True:
    checkpoint = torch.load("./trained_models/resnet18_noTransforms_lr0.001_epochs80")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    num_epochs_trained = checkpoint["epochs"]

model = model.to(device)
print(resnet_model)

epochs = config["epochs"]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, resnet_model, loss_fn, optimizer)
    test_loss = test(test_dataloader, resnet_model, loss_fn)
    if scheduler != None:
        if config["scheduler"] == "ReduceLROnPlateau":
            scheduler.step(test_loss)
        else:
            scheduler.step()
print("Done!")

save = True
if save:
    if load:
        epochs += num_epochs_trained

    epochs=2,
    learning_rate=0.001,
    optimizer="Adam",
    scheduler="ReduceLROnPlateau",
    loss_function="MSELoss",
    image_size="fracture_dataset.csv",
    torch.save({
                'epochs': config["epochs"],
                'model_state_dict': resnet_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'learning_rate': config["learning_rate"],
                'optimizer': config["optimizer"],
                'scheduler': config["scheduler"],
                'loss_function': config["loss_function"],
                'image_size': config["image_size"],
                }, f"./trained_models/resnet18_noTransforms_epochs{epochs}_lr{config['learning_rate']}"
                   f"_optim{config['optimizer']}_sched{config['scheduler']}_loss{config['loss_function']}"
                   f"_imgSize{config['image_size']}")

wandb.run.name = f"./trained_models/resnet18_noTransforms_epochs{epochs}_lr{config['learning_rate']}" \
                 f"_optim{config['optimizer']}_sched{config['scheduler']}_loss{config['loss_function']}" \
                 f"_imgSize{config['image_size']}"
wandb.run.save()
