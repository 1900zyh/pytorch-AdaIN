import argparse
import random
from tqdm import tqdm 
import glob
import os 
import sys
import math

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization, coral


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


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--srcdir', type=str, required=True)
parser.add_argument('--gpuid', type=int, required=True)
parser.add_argument('--style', type=str, default="../wikiart")

# Additional options
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')
parser.add_argument('--content_size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=0,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()
torch.cuda.set_device(int(args.gpuid))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main_worker(): 

    # set up models 
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

    f = open(f"capture_train_GPU{args.gpuid}.txt", "w")
    content_list = glob.glob(os.path.join(args.srcdir, "*/*"))
    
    style_list = glob.glob(os.path.join(args.style, "*/*"))
    content_list.sort()
    style_list.sort()

    chunk = math.ceil(len(content_list) / 7)
    start = max(0, args.gpuid*chunk)
    end = min((args.gpuid+1)*chunk, len(content_list))

    for content_path in tqdm(content_list[start:end]):
        # if do_interpolation:  # one content image, N style image
        #     style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        #     content = content_tf(Image.open(str(content_path))) \
        #         .unsqueeze(0).expand_as(style)
        #     style = style.to(device)
        #     content = content.to(device)
        #     with torch.no_grad():
        #         output = style_transfer(vgg, decoder, content, style,
        #                                 args.alpha, interpolation_weights)
        #     output = output.cpu()
        #     output_name = output_dir / '{:s}_interpolation{:s}'.format(
        #         content_path.stem, args.save_ext)
        #     save_image(output, str(output_name))
        try: 
            style_path = random.choice(style_list)
            content = content_tf(Image.open(str(content_path)).convert("RGB"))
            style = style_tf(Image.open(str(style_path)))
            if args.preserve_color:
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        args.alpha)
            output = output.cpu()

            output_name = "style_" + content_path
            os.makedirs(os.path.dirname(output_name), exist_ok=True)

            save_image(output, output_name)
        except: 
            print(f"skip {content_path} ...")
            f.write(content_path+"\n")
            f.flush()
    f.close()




if __name__ == "__main__": 
    main_worker()