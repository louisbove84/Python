import argparse

"""
        Credit for assistance: https://github.com/cjimti/aipnd-project & https://github.com/DMells/Convolutional-Neural-Networks-Project
"""

def get_args():
    parser = argparse.ArgumentParser(
        description="Identify Image",
        usage="python ./predict.py /path/to/image.jpg checkpoint.pth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument('checkpoint', action="store", nargs='*', default='/home/workspace/ImageClassifier/checkpoint.pth')
    parser.add_argument('path_to_image', action="store", nargs='*', default='/home/workspace/ImageClassifier/flowers/test/1/image_06752.jpg')
    parser.add_argument('--save_dir', action="store", default=".", dest='save_dir')
    parser.add_argument('--top_k', action="store", default=5, dest='top_k')
    parser.add_argument('--category_names', action="store", default="cat_to_name.json", dest='categories_json')
    parser.add_argument('--gpu', action="store_true", dest="use_gpu",  default=False)

    parser.parse_args()
    return parser

def main():
    """
        Main Function
    """
    print(f'Command line argument utility for predict.py.\nTry "python train.py -h".')


if __name__ == '__main__':
    main()
