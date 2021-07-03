#Imports

# Imports here
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
import argparse

parser = argparse.ArgumentParser(description = "Predict Image")

parser.add_argument('-g', '--gpu', default=True, action = 'store_true', help="Default GPU, switch to CPU")
parser.add_argument('-f', '--save_location', default='./checkpoint.pth', help="location of saved checkpoint")
parser.add_argument('-i', '--image', default='flowers/test/13/image_05761.jpg', help="image to predict")
parser.add_argument('-t', '--top_pred', type = str, default = 5, help= "top flower predictions")
parser.add_argument('-m', '--mapping_file', default = 'cat_to_name.json', help="json mapping file")

args = parser.parse_args()

image = args.image
# ## Loading the checkpoint

def checkpoint_load(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    classifier = checkpoint['classifier']
    
    model = getattr(models, checkpoint['model'])(pretrained=True)
    
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
    

device =args.gpu

model.class_to_idx = args.mapping_file


#Prediction
#topk = args.top_pred

def predict(image, model, topk = args.top_pred):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    predimg = Image.open(args.image)
    predimg = process_image(predimg)
    
    predimg = np.expand_dims(predimg, 0)
    
    predimg = torch.from_numpy(predimg)
    
    model.eval()
    inputs = Variable(predimg).to(device)
    logits = model.forward(inputs)
    
    ps = F.softmax(logits, dim=1)
    topk = ps.cpu().topk(topk)
    
    return (e.data.numpy().squeeze().tolist() for e in topk)

#Run predict to get probabilities and indices



image_path = args.image
  
prob,classes = predict(image_path, model)
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

names = list(map(lambda x: cat_to_name[f"{idx_to_class[x]}"], classes))
print(prob)
print(classes)
print(names)


if __name__ == '__main__':
    main()








