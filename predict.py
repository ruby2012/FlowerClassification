#Imports

import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision 
from torchvision import transforms, models, datasets
from torchvision.datasets import ImageFolder 
from collections import OrderedDict
import time
import copy
from PIL import Image
import json
import argparse

parser = argparse.ArgumentParser(description = "Predict Image")

parser.add_argument('-a', '--arch', 
                    choices= ["resnet18","resnet", "vgg", "vgg16"],
                    help = "Training model, resnet18 or vgg16")
parser.add_argument('-g', '--gpu', default=True, action = 'store_true', help="Default GPU, switch to CPU")
parser.add_argument('-f', '--save_location', default='./checkpoint.pth', help="location of saved checkpoint")
parser.add_argument('-i', '--image', default='flowers/test/13/image_05761.jpg', help="image to predict")
parser.add_argument('-t', '--top_pred', type = int, default = 5, help= "top flower predictions")
parser.add_argument('-m', '--mapping_file', default = 'cat_to_name.json', help="json mapping file")
parser.add_argument('-d', '--image_data', default='flowers', help="image data")
parser.add_argument('-u', '--hidden_units', type = int, default=512, help = "number of hidden layers")




args = parser.parse_args()

image = args.image


#mapping file
with open(args.mapping_file, 'r') as f:
    args.mapping_file = json.load(f)
map_to_name = args.mapping_file    
#Load Data
data_dir = args.image_data
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train = transforms.Compose((transforms.RandomRotation(30),
                                           transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])))

validate = transforms.Compose((transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])))

test = transforms.Compose((transforms.RandomRotation(30),
                                           transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])))
# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = train)
validate_dataset = datasets.ImageFolder(valid_dir, transform = validate)
test_dataset =  datasets.ImageFolder(test_dir, transform = test)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=50)
test_loader= torch.utils.data.DataLoader(test_dataset, batch_size=50)

# ## Loading the checkpoint

def checkpoint_load(checkpoint_file):
    checkpoint = torch.load(args.save_location)
    
    if args.arch == 'resnet' or args.arch == 'resnet18':
        model = models.resnet18(pretrained = True)
        input_size = 512 
    elif args.arch == 'vgg' or args.arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        input_size = 25088 
    classifier = checkpoint['classifier']
    model.classifier = nn.Sequential(nn.Linear(input_size, args.hidden_units), 
                           nn.ReLU(),
                           nn.Dropout (p=.25),
                           nn.Linear(args.hidden_units, 102),
                           nn.LogSoftmax(dim=1))
    
    if args.arch == 'resnet' or args.arch == 'resnet18':
        model.fc = classifier
    elif args.arch == 'vgg' or args.arch == 'vgg16':
        model.classifier = classifier
      
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict = False)
        
    return model, checkpoint['class_to_idx']

model,class_to_idx = checkpoint_load('checkpoint.pth')

image = args.image

# TODO: Process a PIL image for use in a PyTorch model

def process_image(image):
    #''' Scales, crops, and normalizes a PIL image for a PyTorch model,
     #   returns an Numpy array'''
   
    image_loader = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    image = image_loader(image)

    
    return image
    

device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

model.class_to_idx = args.mapping_file


#Prediction

def predict(image, model, topk = args.top_pred):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    predimg = Image.open(args.image)
    predimg = process_image(predimg)
    
    predimg = np.expand_dims(predimg, 0)
    
    predimg = torch.from_numpy(predimg)
    
    model.eval()
    model = model.to(device)
    
    inputs = Variable(predimg).to(device)
    logits = model.forward(inputs)
                                      
    ps = F.softmax(logits, dim=1)
    topk = ps.cpu().topk(topk)
    
    return (e.data.numpy().squeeze().tolist() for e in topk)

#Run predict to get probabilities and indices



image_path = args.image
image_name = map_to_name[f"{image_path.split('/')[-2]}"]
  
prob,classes = predict(image_path, model)
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

names = list(map(lambda x: map_to_name[f"{idx_to_class[x]}"], classes))
print(prob)
print(classes)
print(names)










