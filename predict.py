#Imports
import torch
import os, random
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision 
from torchvision import transforms, models, datasets
from torchvision.datasets import ImageFolder 
from collections import OrderedDict
import copy
import time
from PIL import Image

# ## Loading the checkpoint
def checkpoint_load(checkpoint_file): 
    torch.load(checkpoint.pth)
    
    arch = checkpoint['arch']
                      
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    checkpoint = torch.load(checkpoint_file)
    classifier = nn.Sequential(nn.Linear(512, 1024), 
                           nn.ReLU(),
                           nn.Dropout (p=.25),
                           nn.Linear(1024, 102),
                           nn.LogSoftmax(dim=1))
    
    for param in model.parameters(): param.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict = False)
        
    return model


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
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.class_to_idx = checkpoint['class_to_idx']


#Prediction

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    predimg = Image.open(image_path)
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


img = random.choice(os.listdir('flowers/train/13/'))
image_path = 'flowers/train/13/' + img
  
prob,classes = predict(image_path, model)
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

names = list(map(lambda x: cat_to_name[f"{idx_to_class[x]}"], classes))
print(prob)
print(classes)
print(names)











