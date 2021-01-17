import os 
import glob 
import sys 
import argparse 
from tqdm import tqdm
from PIL import Image

import torch
import time
import importlib
import torchvision.models as models 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('-d', '--data', type=str, required=True)
parser.add_argument('-i', '--id', type=int, required=True)
# parser.add_argument('-a', '--arch', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--workers', type=int, default=8)
args = parser.parse_args()



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
class ImageNetDataset(Dataset):
    def __init__(self, datadir, label_file): 
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        self.img_list = list(glob.glob(os.path.join(datadir, "val", "*")))
        self.img_list.sort()
        
        with open(label_file, 'r') as f: 
            labels = f.readlines()
        self.img_labels = [int(l.strip().split()[-1]) for l in labels]
    
    def __len__(self,): 
        return len(self.img_list)
    
    def __getitem__(self, index): 
        img_path = self.img_list[index]
        img_name = os.path.basename(img_path)

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        label = self.img_labels[index]
        return img, label, img_name


def main_worker(arch): 
    
    # set model
    torch.cuda.set_device(int(args.id))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.__dict__[arch](pretrained=True).cuda()
    model.eval()

    # set data 
    valdir = os.path.join(args.data, 'val')
    val_dataset = ImageNetDataset(args.data, "ILSVRC2012_validation_ground_truth.txt")

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True)

    # set metrics
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    batch_time = AverageMeter('Time', ':6.3f')
    # progress = ProgressMeter(
    #     len(val_dataset),
    #     [batch_time, top1, top5],
    #     prefix='Test: ')

    # test 
    with torch.no_grad():
        end = time.time()
        for i, (images, target, name) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % 500 == 0:
            #     progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        return top1.avg, top5.avg
        # print(
        #     f"Test {args.arch} on {args.data}" 
        #     " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        # )


if __name__ == "__main__": 
    arch_list = [
        "alexnet", "vgg16", 
        "resnet18", "resnet50", "resnet101", "resnet152",
        "densenet121", "densenet201",
        "resnext50_32x4d", "resnext101_32x8d"]
    for arch in arch_list: 
        top1, top5 = main_worker(arch)
        print(f"{args.data}-{arch}: Acc@1 {top1:.4f}  Acc@5 {top5:.4f}")