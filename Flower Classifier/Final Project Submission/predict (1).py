import json
import torch
import warnings
import predict_args
import train_args
import argparse
import train

from PIL import Image
from torchvision import models
from torchvision import transforms

def main():
    """
        Image Classification Network Trainer
        Student: Louis Bove
        Credit for assistance: https://github.com/cjimti/aipnd-project & https://github.com/DMells/Convolutional-Neural-Networks-Project
    """
    
    parser = predict_args.get_args()
    cli_args = parser.parse_args()
    
    # Load using CPU and then request GPU
    device = torch.device("cpu")
    if cli_args.use_gpu:
        device = torch.device("cuda:0")

    # Load needed categories and model
    with open(cli_args.categories_json, 'r') as f:
        cat_to_name = json.load(f)
        
    checkpoint_model = load_checkpoint(device, cli_args.checkpoint)

    top_prob, top_classes = predict(cli_args.path_to_image, checkpoint_model, cli_args.top_k)

    label = top_classes[0]
    prob = top_prob[0]

    print(f'Parameters\n')

    print(f'Current image - {cli_args.path_to_image}')
    print(f'Model used - {cli_args.checkpoint}')
    print(f'Device used - {device}')

    print(f'\nPrediction\n')

    print(f'Flower      : {cat_to_name[label]}')
    print(f'Label       : {label}')
    print(f'Probability : {prob*100:.2f}%')

    print(f'\nTop K\n')

    for i in range(len(top_prob)):
        print(f"{cat_to_name[top_classes[i]]:<25} {top_prob[i]*100:.2f}%")


def predict(image_path, model, topk=5):

    # Implement the code to predict the class from an image file
    
    # evaluation mode (https://pytorch.org/docs/stable/nn.html#torch.nn.Module.eval)
    model.eval()
    
    # cpu mode
    model.cpu()
    
    # load image as torch.Tensor
    image = process_image(image_path)
    
    # Unsqueeze will return a tensor with a dimension of size one (https://pytorch.org/docs/stable/torch.html#torch.unsqueeze)
    image = image.unsqueeze(0)
    
    # Disabling gradient calculation 
    with torch.no_grad():
        output = model.forward(image)
        top_prob, top_labels = torch.topk(output, topk)
        
        # Calculate the exponentials
        top_prob = top_prob.exp()
        
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    print(model.class_to_idx)
    print(top_labels.numpy()[0])
    print(class_to_idx_inv[0])
    
    for label in top_labels.numpy()[0]:
        
        mapped_classes.append(class_to_idx_inv[label])
        
    return top_prob.numpy()[0], mapped_classes


# Loading a checkpoint in order to rebuild the model

def load_checkpoint(device, file='checkpoint.pth'):
    # Loading weights for CPU model while trained on GP
    # https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032
    model_state = torch.load(file, map_location=lambda storage, loc: storage)

    model = models.__dict__[model_state['arch']](pretrained=True)
    model = model.to(device)

    model.classifier = model_state['classifier']
    model.load_state_dict(model_state['state_dict'])
    model.class_to_idx = model_state['class_to_idx']

    return model

def process_image(image):
    # This will scale, crop, and normalize an image for the PyTorch model IOT return a Numpy array
    expects_means = [0.485, 0.456, 0.406]
    expects_std = [0.229, 0.224, 0.225]
           
    pil_image = Image.open(image).convert("RGB")
    
    # Utilize transforms to make the nessessary changes to the image
    in_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(expects_means, expects_std)])
    pil_image = in_transforms(pil_image)

    return pil_image


if __name__ == '__main__':
    # some models return deprecation warnings
    # https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
