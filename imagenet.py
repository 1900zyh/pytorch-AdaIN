import argparse
import random
from tqdm import tqdm 
import glob
import os 
import sys
import math
from functools import partial 
import multiprocessing
from multiprocessing import Pool

import torch
import torch.nn as nn
from PIL import Image, PngImagePlugin, ImageFile
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

import net
from function import adaptive_instance_normalization, coral


# update maximal loading size for extreme high-resolution images
Image.MAX_IMAGE_PIXELS = 10000000000000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--srcdir', type=str, required=True)
parser.add_argument('--gpuid', type=int, required=True)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--alpha', type=float, default=0.5,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')

# Additional options
parser.add_argument('--style', type=str, default="../wikiart")
parser.add_argument('--content_size', type=int, default=256,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=256,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')
args = parser.parse_args()


class ImageNetDataset(Dataset):
    def __init__(self, content_list, style_list, content_tf, style_tf): 
        self.content_list = content_list
        self.style_list = style_list
        self.content_tf = content_tf
        self.style_tf = style_tf
    
    def __len__(self,): 
        return len(self.content_list)
    
    def __getitem__(self, index): 
        style_path = random.choice(self.style_list)
        content_path = self.content_list[index]
        content = self.content_tf(Image.open(str(content_path)).convert("RGB"))
        style = self.style_tf(Image.open(str(style_path)))
        if args.preserve_color:
            style = coral(style, content)
        return content, style, content_path


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def main_worker(): 

    # set up models 
    # torch.cuda.set_device(int(args.gpuid))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder = net.decoder
    vgg = net.vgg
    decoder.eval()
    vgg.eval()
    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg.to(device)
    decoder.to(device)

    # set up transformation 
    content_tf = test_transform(args.content_size, args.crop)
    style_tf = test_transform(args.style_size, args.crop)

    content_list = glob.glob(os.path.join(args.srcdir, "*"))
    style_list = glob.glob(os.path.join(args.style, "*/*"))
    content_list.sort()
    style_list.sort()

    gpu_count = 4
    gpuid = args.gpuid%gpu_count
    chunk = math.ceil(len(content_list) / gpu_count)
    start = max(0, gpuid*chunk)
    end = min((gpuid+1)*chunk, len(content_list))

    f = open(f"capture_train_{args.gpuid}.txt", "w")
    image_dataset = ImageNetDataset(
        content_list, style_list, content_tf, style_tf)
    image_loader = DataLoader(
        image_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True)

    for content, style, content_paths in tqdm(image_loader):
        try: 
            style = style.to(device)
            content = content.to(device)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        args.alpha)
            output = output.cpu()
            
            for i in range(len(content_paths)): 
                output_name = f"style{args.alpha}_" + content_paths[i]
                os.makedirs(os.path.dirname(output_name), exist_ok=True)
                save_image(output[i:i+1,...], output_name)
        except: 
            for i in range(len(content_paths)): 
                print(f"skip {content_paths[i]} ...")
                f.write(content_paths[i]+"\n")
                f.flush()
    f.close()


if __name__ == "__main__": 
    main_worker()