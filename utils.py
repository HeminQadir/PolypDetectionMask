import json
from datetime import datetime
from pathlib import Path
import random
import numpy as np
import torch
from torch.autograd import Variable
import tqdm
from loss import L1Loss, GAN_loss, Discrim_loss, KLDivLoss
from torch.optim import Adam

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


def train(args, model, train_loader, n_epochs=None, fold=None):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer_G = Adam(model.parameters(), lr=lr)
    
    criterion_pixelwise = torch.nn.L1Loss() # #torch.nn.MSELoss()

    root = Path(args.root)
    model_path = Path(F"/content/gdrive/My Drive/Pytorch_model/model_{fold}.pt".format(fold=fold))#root / 'model_{fold}.pt'.format(fold=fold)
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
    
    
    model_path_epoch = Path(F"/content/gdrive/My Drive/Pytorch_model/"+str(n_epochs)+"_model_{fold}.pt".format(fold=fold))
    save_epoch = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path_epoch))

    report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        #model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                # Model inputs 
                inputs, targets = variable(inputs), variable(targets)#.long()
                
                # ------------------
                #  Train Generators
                # ------------------
                optimizer_G.zero_grad()
                # Generate output
                outputs  = model(inputs)
              
                # Pixel-wise loss
                loss_pixel = criterion_pixelwise(outputs, targets)
                #loss_pixel1 = criterion_pixelwise(x_out1, targets)
                #loss_pixel2 = criterion_pixelwise(x_out2, targets)
                #loss_pixel3 = criterion_pixelwise(x_out3, targets)
                #loss_pixel4 = criterion_pixelwise(x_out4, targets)
                #loss_pixel5 = criterion_pixelwise(x_out5, targets)

                # Total loss of GAN
                loss_G = 100*loss_pixel #+(loss_pixel1+loss_pixel2+loss_pixel3+loss_pixel4+loss_pixel5)
                
                
                loss_G.backward()
                optimizer_G.step()
                
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
            save_epoch(epoch + 1)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            save_epoch(epoch)
            print('done.')
            return
