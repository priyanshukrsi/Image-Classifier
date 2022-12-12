import torch
from torchvision import datasets, transforms, models
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description = "Parser of training script")

parser.add_argument ('data_dir', help = 'Provide data directory. Mandatory argument', type = str)
parser.add_argument ('--save_dir', help = 'Provide saving directory. Optional argument', type = str)
parser.add_argument ('--arch', help = 'Vgg13 can be used if this argument specified, otherwise Alexnet will be used', type = str)
parser.add_argument ('--lrn', help = 'Learning rate, default value 0.001', type = float)
parser.add_argument ('--hidden_units', help = 'Hidden units in Classifier. Default value is 2048', type = int)
parser.add_argument ('--epochs', help = 'Number of epochs', type = int)
parser.add_argument ('--GPU', help = "Option to use GPU", type = str)

args = parser.parse_args ()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


if args.GPU == 'GPU' and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

data_transforms_train = transforms.Compose([transforms.RandomRotation (30),
                                            transforms.RandomResizedCrop (224),
                                            transforms.RandomHorizontalFlip (),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

data_transforms_test = transforms.Compose([transforms.Resize (255),
                                           transforms.CenterCrop (224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

data_transforms_validate = transforms.Compose([transforms.Resize (255),
                                               transforms.CenterCrop (224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets_train = datasets.ImageFolder (train_dir, transform = data_transforms_train)
image_datasets_test = datasets.ImageFolder (train_dir, transform = data_transforms_test)
image_datasets_validate = datasets.ImageFolder (train_dir, transform = data_transforms_validate)


# TODO: Using the image datasets and the trainforms, define the dataloaders
loader_train = torch.utils.data.DataLoader(image_datasets_train, batch_size = 64, shuffle = True)
test_loader_test = torch.utils.data.DataLoader(image_datasets_test, batch_size = 64, shuffle = True)
tloader_validate = torch.utils.data.DataLoader(image_datasets_validate, batch_size = 64, shuffle = True)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def load_model (arch, hidden_units):
    if arch == 'vgg13': #setting model based on vgg13
        model = models.vgg13 (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units: #in case hidden_units were given
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        else: #if hidden_units not given
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (25088, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))
    else: #setting model based on default Alexnet ModuleList
        arch = 'alexnet' #will be used for checkpoint saving, so should be explicitly defined
        model = models.alexnet (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units: #in case hidden_units were given
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (9216, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        else: #if hidden_units not given
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (9216, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))
    model.classifier = classifier #we can set classifier only once as cluasses self excluding (if/else)
    return model, arch


def validation(model, valid_loader, criterion):
    model.to (device)
    
    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

model, arch = load_model (args.arch, args.hidden_units)

criterion = nn.NLLLoss ()
if args.lrn: #if learning rate was provided
    optimizer = optim.Adam (model.classifier.parameters (), lr = args.lrn)
else:
    optimizer = optim.Adam (model.classifier.parameters (), lr = 0.001)


model.to(device) #device can be either cuda or cpu
#setting number of epochs to be run
if args.epochs:
    epochs = args.epochs
else:
    epochs = 5

print_every = 40
steps = 0


for e in range (epochs):
    running_loss = 0
    for images,labels in loader_train:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad () 
        outputs = model.forward(images) 
        loss = criterion (outputs, labels) 
        loss.backward ()
        optimizer.step () 
        running_loss += loss.item () 

        if steps % print_every == 0:
            model.eval () 
            with torch.no_grad():
                valid_loss, accuracy = validation(model, tloader_validate, criterion)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(tloader_validate)),
                  "Valid Accuracy: {:.3f}%".format(accuracy/len(tloader_validate)*100))

            running_loss = 0
            model.train()


model.to ('cpu') 
model.class_to_idx = image_datasets_train.class_to_idx 

checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': arch,
              'mapping':    model.class_to_idx
             }

if args.save_dir:
    torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save (checkpoint, 'checkpoint.pth')
