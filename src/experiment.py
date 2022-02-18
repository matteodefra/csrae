from __future__ import print_function
import time

import torch
from torch.autograd import Variable

import numpy as np

from tqdm import tqdm

from torchviz import make_dot

from torchvision.utils import save_image

import math

from vae import StandardVAE
from csrae import CSRAE

import pathlib

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================

def fit_model(EPOCHS, DEVICE, train_loader, valid_loader, model, optimizer, decay):

    name_dir_models = "models_CSRAE" if isinstance(model, CSRAE) else "models_VAE"

    pathlib.Path(name_dir_models).mkdir(parents=True, exist_ok=True) 

    validation_steps = 0
    best_model = None

    max_valid_loss = math.inf
    train_loss, train_re, train_divergence, valid_loss, valid_re, valid_divergence = [], [], [], [], [], []

    for epoch in range(EPOCHS):

        print("Epoch : {}/{}".format(epoch+1, EPOCHS))

        start = time.time()
        train_vae(DEVICE, train_loader, model, optimizer, train_loss, train_re, train_divergence)
        valid_vae(DEVICE, valid_loader, model, valid_loss, valid_re, valid_divergence)
        end = time.time()

        print("Time elapsed: {}".format(end-start))

        if isinstance(model, CSRAE):
            print("Training Losses: {:.5f} Reconstruction Losses: {:.5f} CS divergence: {:.5f}".format(np.mean(train_loss), np.mean(train_re), np.mean(train_divergence)))
            print("Validation Losses: {:.5f} Reconstruction Losses: {:.5f} CS divergence: {:.5f}".format(np.mean(valid_loss), np.mean(valid_re), np.mean(valid_divergence)))

        else:
            print("Training Losses: {:.5f} Reconstruction Losses: {:.5f} KL divergence: {:.5f}".format(np.mean(train_loss), np.mean(train_re), np.mean(train_divergence)))
            print("Validation Losses: {:.5f} Reconstruction Losses: {:.5f} KL divergence: {:.5f}".format(np.mean(valid_loss), np.mean(valid_re), np.mean(valid_divergence)))

        # decay.step(np.mean(valid_loss))

        # checkpoint based on validation accuracy
        if valid_loss[-1] < max_valid_loss:
            validation_steps = 0
            max_valid_loss = valid_loss[-1]
            best_model = model
            torch.save({'model': model, 'optimizer': optimizer}, f"{name_dir_models}/model_{epoch}_{max_valid_loss:.3f}.tar")
            print(f'Save model epoch {epoch}, validation loss {valid_loss[-1]}...') 
        else:
            validation_steps += 1
            if validation_steps == 100:
                # Early stopping heuristics
                return model, train_loss, train_re, train_divergence, valid_loss, valid_re, valid_divergence, best_model

    return model, train_loss, train_re, train_divergence, valid_loss, valid_re, valid_divergence, best_model



def train_vae(DEVICE, train_loader, model, optimizer, train_loss, train_re, train_divergence):
    # set model in training mode
    model.train()

    print("Training")

    # with torch.autograd.detect_anomaly():
    for img, label in tqdm(train_loader):

        img = img.to(DEVICE)
        label = label.to(DEVICE)
        pred, mu, log_var = model(img)

        total_loss, recons_loss, kld_loss = model.loss_function(pred, img, mu, log_var)

        train_loss.append(total_loss.item())
        train_re.append(recons_loss.item())
        train_divergence.append(kld_loss.item())

        optimizer.zero_grad()
        total_loss.backward()

        optimizer.step()


def valid_vae(DEVICE, valid_loader, model, valid_loss, valid_re, valid_divergence):
    
    model.eval()

    print("Validation")

    with torch.no_grad():
        for img, label in tqdm(valid_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred, mu, log_var = model(img)
            total_loss, recons_loss, kld_loss = model.loss_function(pred, img, mu, log_var)

            valid_loss.append(total_loss.item())
            valid_re.append(recons_loss.item())
            valid_divergence.append(kld_loss.item())


def reconstruction(DEVICE, test_loader, model, tries):

    name_dir_images = "images_CSRAE" if isinstance(model,CSRAE) else "images_VAE"

    pathlib.Path(name_dir_images).mkdir(parents=True, exist_ok=True) 

    model.eval()

    with torch.no_grad():
        for i in range(tries):
            imgs, _ = next(iter(test_loader))
            imgs = imgs.to(DEVICE)
            recons_datas = model.generate(imgs)

            if isinstance(model,CSRAE):
                save_image(imgs.view(32, 3, 32, 32)[:25],
                            f"{name_dir_images}/Original_image_{i}.png", nrow=5, normalize=True) 
                save_image(recons_datas.view(32, 3, 32, 32)[:25],
                            f"{name_dir_images}/CSRAE_Reconstruction_{i}.png", nrow=5, normalize=True)
            else:
                save_image(imgs.view(32, 3, 32, 32)[:25],
                            f"{name_dir_images}/Original_image_{i}.png", nrow=5, normalize=True) 
                save_image(recons_datas.view(32, 3, 32, 32)[:25],
                            f"{name_dir_images}/VAE_Reconstruction_{i}.png", nrow=5, normalize=True)


def sampling(model, tries, DEVICE):

    name_dir_images = "sampling_CSRAE" if isinstance(model,CSRAE) else "sampling_VAE"

    pathlib.Path(name_dir_images).mkdir(parents=True, exist_ok=True) 

    num_samples = 25

    with torch.no_grad():   
        for i in range(tries):
            samples = model.sample(num_samples, DEVICE)
            if isinstance(model,CSRAE):
                save_image(samples.view(25, 3, 32, 32)[:25], f"{name_dir_images}/CSRAE_Sampling_{i}.png",
                                    nrow=5, normalize=True)
            else:
                save_image(samples.view(25, 3, 32, 32)[:25], f"{name_dir_images}/VAE_Sampling_{i}.png",
                                    nrow=5, normalize=True)

