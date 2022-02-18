from random import shuffle
from torch.utils.data import DataLoader

from torchvision.utils import save_image
from torchvision import datasets, transforms

import torch

import numpy as np

import argparse
from experiment import fit_model, reconstruction, sampling
from csrae import CSRAE

from vae import StandardVAE
from torch.utils.data.sampler import SubsetRandomSampler


# Take input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', help='path/to/train/data')
parser.add_argument('-hd', '--hidden', type=int, help='number of hidden unit')
parser.add_argument('-ld', '--latent', type=int, help="number of latent unit")
parser.add_argument('-lr', '--learning', type=int, help="learning rate")
parser.add_argument('-e', '--epochs', type=int, help='epochs')
parser.add_argument('-b', '--batch_size', type=int,help='Batch size')
parser.add_argument('-m', '--model', help="model type")
parser.add_argument('-K', '--K', type=int, help="Piors")
parser.add_argument('-mp', '--model_path', help="path to load saved model from")
parser.add_argument('-nt', "--number_testing", type=int, help="number of training/sampling executions")

args = vars(parser.parse_args())


DATA_PATH = ("./" + args["data"] if args["data"] else "")
HIDDEN = (args["hidden"] if args["hidden"] else 32)
LATENT = (args["latent"] if args["latent"] else 2)
LR = (args["learning"] if args["learning"] else 1e-4)
BATCH_SIZE = (args["batch_size"] if args["batch_size"] else 32)
EPOCHS = (args["epochs"] if args["epochs"] else 30)
MODEL = (args["model"] if args["model"] else "standard")
K = (args["K"] if args["K"] else 50)
MODEL_PATH = (args["model_path"] if args["model_path"] else "" )
TRIES = (args["number_testing"] if args["number_testing"] else 10)

train_transform = transforms.Compose([ 
    transforms.ToTensor()])

# data augmentation
transform_train = transforms.Compose([
    transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.4),
    transforms.RandomApply([transforms.RandomAffine(0, scale=(0.8,1.2))], 0.4),
    transforms.RandomApply([transforms.RandomAffine(10)], p=0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize([0.], [1.]),
])
transform_valid = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.], [1.]),
])

validation_split = .2
random_seed = 42

dataset = datasets.CIFAR10(root=DATA_PATH, train = True, transform = transform_train, download=True)

print(dataset.classes)

print(len(dataset))

dataset_size = len(dataset)

indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if True:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, 
                                           sampler=train_sampler, drop_last=True)
validation_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                                sampler=valid_sampler, drop_last=True)


test_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)

print(len(test_dataset))

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if MODEL == "standard":
    model = StandardVAE(input_shape = (3,32, 32)).to(DEVICE)
else:
    model = CSRAE(input_shape=(3, 32, 32), K=K).to(DEVICE)

# print(model)

optim = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

if not MODEL_PATH:

    model, total_training_losses, total_training_reconstruction, total_training_divergence, total_valid_losses, total_valid_re, total_valid_divergence, best_model = fit_model(EPOCHS, DEVICE, train_loader, validation_loader, model, optim, scheduler)

    print(total_training_losses)
    print(total_valid_losses)

else:
    checkpoint = torch.load(MODEL_PATH)
    model = (checkpoint["model"])

    print(model)

    model.eval()

reconstruction(DEVICE, test_loader, model, TRIES)

sampling(model, TRIES, DEVICE)

# """
#     Reconstruction
# """
# imgs, _ = next(iter(train_loader))
# imgs = imgs.to(DEVICE)
# recons_datas = model.generate(imgs)
# print(recons_datas.size())
# save_image(imgs.view(32, 3, 32, 32)[:25],
#             "Original_image.png", nrow=5, normalize=True) 
# save_image(recons_datas.view(32, 3, 32, 32)[:25],
#             "Standard_VAE_Reconstruction.png", nrow=5, normalize=True)

# """
#     Sampling
# """
# num_samples = 25

# samples = model.sample(num_samples)
# save_image(samples.view(25, 3, 32, 32)[:25], "Standard_VAE_Sampling.png",
#             nrow=5, normalize=True)