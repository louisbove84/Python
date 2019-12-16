import argparse

# The first file, train.py, will train a new network on a dataset and save the model as a checkpoint

# Train a new network on a data set with train.py

# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 9
# Use GPU for training: python train.py data_dir --gpu

"""
        Credit for assistance: https://github.com/cjimti/aipnd-project & https://github.com/DMells/Convolutional-Neural-Networks-Project
"""



# List out the supported nn architecture
nn_arch = [
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
]

def get_args():
    parser = argparse.ArgumentParser(
        description="Train and save an image classification model.",
        usage="python ./train.py ./flowers/train --gpu --learning_rate 0.001 --hidden_units 512 --epochs 9",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('data_directory', action="store", nargs='*', default='/home/workspace/ImageClassifier/flowers/valid')
    parser.add_argument('--save_dir', action="store", default=".", dest='save_dir', type=str)
    parser.add_argument('--save_name', action="store", default="checkpoint", dest='save_name')
    parser.add_argument('--categories_json', action="store", default="cat_to_name.json", dest='categories_json')
    parser.add_argument('--arch', action="store", default="vgg16", dest='arch')
    parser.add_argument('--gpu', action="store_true", dest="use_gpu", default=False)
    hp = parser.add_argument_group('hyperparameters')
    hp.add_argument('--learning_rate', action="store", default=0.001, type=float,)
    hp.add_argument('--hidden_units', '-hu', action="store", dest="hidden_units", default=[512])
    hp.add_argument('--epochs', action="store", dest="epochs",  default=9)
    parser.parse_args()
    return parser

def main():
    """
        Main Function
    """
    print(f'Command line argument utility for train.py.\nTry "python train.py -h".')


if __name__ == '__main__':
    main()
