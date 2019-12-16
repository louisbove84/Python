import sys
import os
import json
import torch
import predict_args
import train_args
import argparse

from torchvision import datasets, transforms
from collections import OrderedDict
from torchvision import models
from torch.utils.data import DataLoader
from torch import nn, optim

def main():

    """
        Image Classification Network Trainer
        Student: Louis Bove
        Credit for assistance: https://github.com/cjimti/aipnd-project & https://github.com/DMells/Convolutional-Neural-Networks-Project
    """

    parser = train_args.get_args()
    cli_args = parser.parse_args()
    
    # checking for data directory
    if not os.path.isdir(cli_args.data_directory):
        print(f'Data directory {cli_args.data_directory} was not found.')
        exit(1)

    # checking for save directory and then making it if not already created
    if not os.path.isdir(cli_args.save_dir):
        print(f'Directory {cli_args.save_dir} does not exist. Making directory, stand by.')
        os.makedirs(cli_args.save_dir)
        
    # loading categories
    with open(cli_args.categories_json, 'r') as f:
        cat_to_name = json.load(f)

    # Base output on catgory number
    output_size = len(cat_to_name)

    # Define batch size
    batch_size = 32

    # Utilizing transform function to modify image
    training_trans = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    # Defining data set and data loader
    training_dataset = datasets.ImageFolder(cli_args.data_directory, transform=training_trans)

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    # Check nn are only vgg
    if not cli_args.arch.startswith("vgg"):
        print("Program only supports VGG")
        exit(1)
    
    # 
    print(f"Running {cli_args.arch} network.")
    nn_model = models.__dict__[cli_args.arch](pretrained=True)

    input_size = 0

    # Input size for VGG
    if cli_args.arch.startswith("vgg"):
        input_size = nn_model.classifier[0].in_features

    # IOT prevent backpropigation
    for param in nn_model.parameters():
    	param.requires_grad = False 
    
    # Create hidden sizes based on input size
    hidden_sizes = cli_args.hidden_units
    hidden_sizes.insert(0, input_size)

    # Ensure the order is kept (https://pymotw.com/3/collections/ordereddict.html)
    od = OrderedDict()    

    for i in range(len(hidden_sizes) - 1):
        od['fc' + str(i + 1)] = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
        od['relu' + str(i + 1)] = nn.ReLU()
        od['dropout' + str(i + 1)] = nn.Dropout(p=0.15)

    od['output'] = nn.Linear(hidden_sizes[i + 1], output_size)
    od['softmax'] = nn.LogSoftmax(dim=1)

    classifier = nn.Sequential(od)

    # Replace classifier (https://pytorch.org/docs/stable/cuda.html)
    nn_model.classifier = classifier

    # Zero out parameters (https://discuss.pytorch.org/t/zero-grad-optimizer-or-net/1887/5)
    nn_model.zero_grad()

    # The negative log likelihood loss as criterion.
    criterion = nn.NLLLoss()

    # Only train the classifier parameters (https://pytorch.org/docs/stable/optim.html)
    print(f"Optimizer learn rate - {cli_args.learning_rate}.")
    optimizer = optim.Adam(nn_model.classifier.parameters(), lr=cli_args.learning_rate)

    # Start with CPU
    device = torch.device("cpu")

    # Requested GPU
    if cli_args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("GPU is not available. Only CPU available.")

    # Send model to device
    nn_model = nn_model.to(device)

    chk_every = 1

    data_set_len = len(training_dataloader.batch_sampler)

    print(f'Training on {data_set_len} batches of {training_dataloader.batch_size}.')
    print(f'Displaying average loss and accuracy for epoch every {chk_every} batches.')

    for e in range(cli_args.epochs):
        e_loss = 0
        prev_chk = 0
        correct = 0
        total = 0
        print(f'\nEpoch {e+1} of {cli_args.epochs}\n----------------------------')
        for ii, (images, labels) in enumerate(training_dataloader):
            # Move images and labeles 
            images = images.to(torch.device("cuda:0"))
            labels = labels.to(torch.device("cuda:0"))
        
            # Zero out gradients 
            optimizer.zero_grad()
              
            # Propigate both forward and backward 
            outputs = nn_model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            # Running total
            e_loss += loss.item()
        
            # Accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
            # Running total
            itr = (ii + 1)
            if itr % chk_every == 0:
                avg_loss = f'avg. loss: {e_loss/itr:.4f}'
                acc = f'accuracy: {(correct/total) * 100:.2f}%'
                print(f'  Batche {prev_chk:03}-{itr:03}: {avg_loss}, {acc}.')
                prev_chk = (ii + 1)

    print('Training Complete')

    nn_model.class_to_idx = training_dataset.class_to_idx
    model_state = {
        'epoch': cli_args.epochs,
        'state_dict': nn_model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'classifier': nn_model.classifier,
        'class_to_idx': nn_model.class_to_idx,
        'arch': cli_args.arch
    }

    save_location = f'{cli_args.save_dir}/{cli_args.save_name}.pth'
    print(f"Checkpoint saved to {save_location}")

    torch.save(model_state, save_location)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
