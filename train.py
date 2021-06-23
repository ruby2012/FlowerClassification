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
import argparse

########
#Image Classifier 

# parser and arguments

parser = argparse.ArgumentParser(description = "Image Classifier Project")

parser.add_argument('-a', '--arch', default="resnet18", help = "Training model, resnet18 or vgg16")
parser.add_argument('-l', '--learning_rate', type = float, default=".003", help = "model learning rate")
parser.add_argument('-H', '--hidden_units', type = int, default=512, help = "number of hidden layers")
parser.add_argument('-e', '--epochs', type=int, default=3, help="How many training epochs do you want to use")
parser.add_argument('-b', '--batch', type=int, default=50, help="Batch size for training")

args = parser.parse_args()


#Load Data
data_dir = 'flowers'
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

batch = 25
# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch)
test_loader= torch.utils.data.DataLoader(test_dataset, batch_size=batch)


# Label mapping

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# # Building and training the classifier
# use gpu or device if unavailable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: Build and train your network
model = models.resnet18(pretrained = True)

#Turn off model gradients
for param in model.parameters():
    param.requires_grad = False
    
#Define new classifier    
classifier = nn.Sequential(nn.Linear(512, 1024), 
                           nn.ReLU(),
                           nn.Dropout (p=.25),
                           nn.Linear(1024, 102),
                           nn.LogSoftmax(dim=1))
model.fc = classifier

criterion = nn.NLLLoss()

learning_rate = .003
optimizer = optim.Adam(model.fc.parameters(), lr= learning_rate)

model.to(device)

#Train Model
epochs = 3 
steps = 0
running_loss = 0
print_every = 50

print("Training started...")
start_time = time.time()

for epoch in range(epochs):
    for images, labels in train_loader:
        steps += 1
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0 :
            model.eval()
            test_loss = 0
            accuracy = 0
            
            for images, labels in validate_loader:
                images, labels = images.to(device), labels.to(device)
                
                logps = model(images)
                loss = criterion(logps, labels)
                test_loss += loss.item()
                
                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1, dim =1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                
            print(f"Epoch {epoch+1}/{epochs}..",
                  f"Train loss: {running_loss/print_every:.3f}..",
                  f"Test loss: {test_loss/len(test_loader):.3f}..",
                  f"Test accuracy: {accuracy/len(test_loader):.3f}")
            
            running_loss = 0
            model.train()
            end_time = time.time()
            
print('Training ended')
training_time = end_time - start_time
print('Training time: {:.0f}m {:.0f}s'.format(training_time/60, training_time % 60))

#Testing your network

test_loss = 0
test_accuracy = 0
model.eval()

print('Validation started')
start_time = time.time()

for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    
    logps = model.forward(inputs)
    loss = criterion(logps, labels)
    
    test_loss = test_loss + loss.item()
    
    probabilities = torch.exp(logps)
    top_probability, top_class = probabilities.topk(1, dim=1)
    
    equals = top_class ==labels.view(*top_class.shape)
    
    test_accuracy = test_accuracy + torch.mean(equals.type(torch.FloatTensor)).item()
    
end_time = time.time()
print('Validation ended')
validation_time = end_time - start_time
print('Validation time: {:.0f}m {:.0f}s'.format(validation_time/60, validation_time % 60))

print('\nTest:\n Loss:  {:.4f} '.format(test_loss/len(test_loader)),
       'Accuracy: {:.4f}'.format(test_accuracy/len(test_loader)))


# # Save the checkpoint
model.class_to_idx = train_dataset.class_to_idx

checkpoint = {'network' :' resnet18',
              'learning_rate':'.003',
              'epochs':'3',
              'classifier': 'classifier',
              'optimizer':'optimizer.state.dict()',
    
}

torch.save(checkpoint, 'checkpoint.pth')


# ## Loading the checkpoint
def checkpoint_load():    
    checkpoint = torch.load('my_checkpoint.pth')
    
    model = models.resnet18(pretrained=True);
    
    for param in model.parameters(): param.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
        
    return model

               
              

