import argparse
import json
from pathlib import Path
import torch
from torch import nn

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
from models import UNet, UNet11, UNet16, AlbuNet34, MDeNet, EncDec, hourglass, MDeNetplus
from dataset import Polyp
import utils
from prepare_train_val import get_split
from transforms import (DualCompose,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        Rotate,
                        CropCVC612,
                        img_resize,
                        Zoomin,
                        Rescale, 
                        RandomCrop,
                        RandomHueSaturationValue,
                        VerticalFlip)

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=0.3, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=1)
    arg('--limit', type=int, default=10000, help='number of images in epoch')
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.001)
    arg('--workers', type=int, default=12)
    arg('--model', type=str, default='UNet', choices=['UNet', 'UNet11', 'LinkNet34', 'UNet16', 'AlbuNet34', 'MDeNet', 'EncDec', 'hourglass', 'MDeNetplus'])

    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    num_classes = 1
    if args.model == 'UNet':
        model = UNet(num_classes=num_classes)
    elif args.model == 'UNet11':
        model = UNet11(num_classes=num_classes, pretrained=True)
    elif args.model == 'UNet16':
        model = UNet16(num_classes=num_classes, pretrained=True)
    elif args.model == 'MDeNet':
        print('Mine MDeNet..................')
        model = MDeNet(num_classes=num_classes, pretrained=True)
    elif args.model == 'MDeNetplus':
        print('load MDeNetplus..................')
        model = MDeNetplus(num_classes=num_classes, pretrained=True)
    elif args.model == 'EncDec':
        print('Mine EncDec..................')
        model = EncDec(num_classes=num_classes, pretrained=True)
    elif args.model == 'GAN':
        model = GAN(num_classes=num_classes, pretrained=True)
    elif args.model == 'AlbuNet34':
        model = AlbuNet34(num_classes=num_classes, pretrained=False)
    elif args.model == 'hourglass':
        model = hourglass(num_classes=num_classes, pretrained=True) 
    else:
        model = UNet(num_classes=num_classes, input_channels=3)

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model).cuda()   #  nn.DataParallel(model, device_ids=device_ids).cuda()
    
    cudnn.benchmark = True
    
    def make_loader(file_names, shuffle=False, transform=None, limit=None):
        return DataLoader(
            dataset=Polyp(file_names, transform=transform, limit=limit),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_split(args.fold)

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))
    
    train_transform = DualCompose([
        CropCVC612(),
        img_resize(512),
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),
        Rescale(), 
        Zoomin(),
        ImageOnly(RandomHueSaturationValue()),
        ImageOnly(Normalize())
    ])

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform, limit=args.limit)

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    utils.train(
        args=args,
        model=model,
        train_loader=train_loader,
        fold=args.fold
    )


if __name__ == '__main__':
    main()
