import argparse
import json
from pathlib import Path
import torch
import numpy as np
import random
import tqdm
from datetime import datetime
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
from models import AlbuNet34, MDeNet
from GAN import discriminator
from dataset import Polyp
from torch.optim import Adam
from loss import GAN_loss, Discrim_loss
from prepare_train_val import get_split
from transforms import (DualCompose,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        Rotate,
                        CropCVC612,
                        img_resize,
                        Zoomin,
                        Zoomout,
                        Rescale, 
                        RandomHueSaturationValue,
                        VerticalFlip)


def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))


def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()
    
    
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
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=12)
    arg("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    arg("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    # Loss functions
    criterion_GAN = GAN_loss(gan_weight=1)            #torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    criterion_discrim = Discrim_loss(dircrim_weight=1)
    
    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100

    # Initialize generator and discriminator
    model = AlbuNet34(num_classes=1, pretrained=True)
    discrim_model = discriminator() 
    
    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
        discrim_model = nn.DataParallel(discrim_model, device_ids=device_ids).cuda()
       
    # Load pretrained models
    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=args.fold)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))
    
    
    # Optimizers
    optimizer_G = Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = Adam(discrim_model.parameters(), lr=args.lr, betas=(args.b1, args.b2))
       
    # Configure dataloaders
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
    
    report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=args.fold)).open('at', encoding='utf8')
    
    for epoch in range(epoch, args.n_epochs + 1):
            model.train()
            discrim_model.train()
            random.seed()
            tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
            tq.set_description('Epoch {}, lr {}'.format(epoch, args.lr))
            losses = []
            tl = train_loader
            try:
                mean_loss = 0
                for i, (inputs, targets) in enumerate(tl):
                    # Model inputs 
                    inputs, targets = variable(inputs), variable(targets)

                    # ------------------
                    #  Train Generators
                    # ------------------
                    optimizer_G.zero_grad()
                    # Generate output 
                    outputs = model(inputs)
                    # fake loss
                    predict_fake = discrim_model(inputs, outputs)

                    # Pixel-wise loss
                    loss_pixel = criterion_pixelwise(outputs, targets)
                    # Generator loss 
                    loss_GAN = criterion_GAN(predict_fake)
                    # Total loss of GAN
                    loss_G =  loss_GAN + lambda_pixel * loss_pixel

                    loss_G.backward()
                    optimizer_G.step()

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    optimizer_D.zero_grad()
                    # Real loss
                    predict_real = discrim_model(inputs, targets)
                    predict_fake = discrim_model(inputs, outputs.detach())

                    # Discriminator loss 
                    loss_D = criterion_discrim(predict_real, predict_fake)
                    loss_D.backward()
                    optimizer_D.step()

                    step += 1
                    batch_size = inputs.size(0)
                    tq.update(batch_size)
                    losses.append(float(loss_G.data))
                    mean_loss = np.mean(losses[-report_each:])
                    tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                    if i and i % report_each == 0:
                        write_event(log, step, loss=mean_loss)
                write_event(log, step, loss=mean_loss)
                tq.close()
                save(epoch + 1)

            except KeyboardInterrupt:
                tq.close()
                print('Ctrl+C, saving snapshot')
                save(epoch)
                print('done.')
                return



if __name__ == '__main__':
    main()
