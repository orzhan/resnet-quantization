import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
import argparse
import os
import shutil
import time

from model import resnet20

config = {
    'epochs': 200,
    'batch_size': 128,
    'workers': 2,
    'weight_decay':1e-4,
    'momentum':0.9,
    'lr':1e-1,
    'device': 'cpu',
    'do_train': True 
}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=config['batch_size'], shuffle=True,
    num_workers=config['workers'], pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=128, shuffle=False,
    num_workers=config['workers'], pin_memory=True)

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss()
best_prec1=0
device = config['device']

model = resnet20().to(device)
optimizer = torch.optim.SGD(model.parameters(), config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
#optimizer = torch.optim.Adam(model.parameters(), config['lr'], weight_decay=config['weight_decay'])

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=-1)


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input_var = input.to(device)
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, max_batches=None):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))
            if max_batches is not None:
              if i >= max_batches:
                break

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    try:
      wandb.log({
          "Test Accuracy": top1.avg,
          "Test Loss": losses.avg})
    except: 
        pass

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

save_dir = 'checkpoints'
os.mkdir(save_dir)

if config['do_train']:
  wandb.init()
  wandb.watch_called = False 
  wandb.config = config
  torch.manual_seed(42)
  torch.backends.cudnn.deterministic = True
  wandb.watch(model, log="all")

if config['do_train']:
  for epoch in range(0, config['epochs']):
          # train for one epoch
          print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
          train(train_loader, model, criterion, optimizer, epoch)
          lr_scheduler.step()

          # evaluate on validation set
          prec1 = validate(val_loader, model, criterion)

          # remember best prec@1 and save checkpoint
          is_best = prec1 > best_prec1
          best_prec1 = max(prec1, best_prec1)

          if epoch > 0 and epoch % 10 == 0:
              save_checkpoint({
                  'epoch': epoch + 1,
                  'state_dict': model.state_dict(),
                  'best_prec1': best_prec1,
              }, is_best, filename=os.path.join(save_dir, 'checkpoint.th'))

          save_checkpoint({
              'state_dict': model.state_dict(),
              'best_prec1': best_prec1,
          }, is_best, filename=os.path.join(save_dir, 'model.th'))


