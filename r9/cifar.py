import argparse
import os
import torch.nn as nn
import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as T
import torchvision
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import wandb

parser = argparse.ArgumentParser(description='CIFAR-10')
parser.add_argument('--name', default=Path(__file__).stem, type=str, help='job name')
parser.add_argument('--download', action='store_true', default='try to download the dataset')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'mps', help='device')
parser.add_argument('--num-workers', default=16, type=int, help='DataLoader worker processes')
parser.add_argument('--seed', default=3407, help='random seed')
parser.add_argument('--datadir', default=Path.home() / 'data', help='dataset directory')
parser.add_argument('--learning-rate', default=0.1, help='learning_rate')
parser.add_argument('--job-id', default=os.environ.get('SLURM_JOB_ID', 'base'), help='job id')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
parser.add_argument('--epochs', default=35, type=int, help='number of training set iterations')
parser.add_argument('--batch-size', default=256, type=int, help='batch size')
parser.add_argument('--momentum', default=0.9, type=int, help='SGD momentum')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--validation-from-train', default=False, type=bool, help='split validation from training set')

normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.ConvertImageDtype(torch.float),
    normalize,
    T.RandomErasing(),
])
test_transform = T.Compose([
    T.ToTensor(),
    T.ConvertImageDtype(torch.float),
    normalize,
])


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBlock2D(in_channels, out_channels, stride=stride)
        self.conv2 = ConvBlock2D(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        return x + self.conv2(x)


class GlobalPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #return nn.functional.max_pool2d(x, kernel_size=x.size()[2:]).squeeze(-1).squeeze(-1)
        return torch.amax(x, dim=(-2,-1))


def make_model():
    model = nn.Sequential(
        ConvBlock2D(3, 64),
        ConvBlock2D(64, 128),
        nn.MaxPool2d(kernel_size=3, stride=2),
        ResBlock(128, 128),
        ConvBlock2D(128, 256),
        nn.MaxPool2d(kernel_size=3, stride=2),
        ConvBlock2D(256, 256),
        nn.MaxPool2d(kernel_size=3, stride=2),
        ResBlock(256, 256),
        GlobalPool(),
        nn.Linear(256, 10)
    )
    return model


def train_one_epoch(model, criterion, optimizer, loader, *, epoch, scaler, scheduler):
    model.train()

    total_loss = 0
    total_correct = 0
    device = next(model.parameters()).device
    with tqdm(loader, desc=f'train {epoch}') as t:
        for x, y in t:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
            else:
                y_hat = model(x)
                loss = criterion(y_hat, y)
            wandb.log({'train/loss': loss.item(),
                       'train/lr': scheduler.get_last_lr()[0]})
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_correct += (y_hat.argmax(dim=1) == y).sum().item()
            t.set_postfix(loss=loss.item())

            scheduler.step()

    n = len(loader) * loader.batch_size
    reports = {'train/loss': total_loss / len(loader),
               'train/accuracy': total_correct / n}
    wandb.log(reports)
    return reports


def evaluate(model, criterion, loader, *, epoch):
    model.eval()
    total_loss = 0
    total_correct = 0
    device = next(model.parameters()).device
    with tqdm(loader, desc=f'evaluate {epoch}') as t:
        for x, y in t:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            total_loss += loss.item()
            total_correct += (y_hat.argmax(dim=1) == y).sum().item()
        n = len(loader) * loader.batch_size
        reports = {'valid/loss': total_loss / len(loader),
                   'valid/accuracy': total_correct / n,
                   'valid/epoch': epoch}
        wandb.log(reports)
        t.set_postfix(**reports)
    return reports


def test(model, loader):
    model.eval()
    total_correct = 0
    device = next(model.parameters()).device

    flip = T.RandomHorizontalFlip()
    crop = T.RandomCrop(32, padding=4)
    erase = T.RandomErasing()

    with tqdm(loader, desc='test') as t:
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            x_flip = flip(x)
            y_hat_flip = model(x_flip)

            x_crop = crop(x)
            y_hat_crop = model(x_crop)

            x_erase = erase(x)
            y_hat_erase = model(x_erase)

            total_correct += (y_hat.argmax(dim=1) == y).sum().item()

        n = len(loader) * loader.batch_size
        reports = {'test/accuracy': total_correct / n}
        wandb.log(reports)
        t.set_postfix(**reports)
        t.refresh()
        return reports


if __name__ == '__main__':
    args = parser.parse_args()
    logger.info('{}', args)

    wandb.init(project="cifar10-base", config=vars(args), save_code=True)

    torch.manual_seed(args.seed)

    if args.validation_from_train:
        train = torchvision.datasets.CIFAR10(args.datadir, transform=transform, train=True, download=args.download)
        valid = torchvision.datasets.CIFAR10(args.datadir, transform=test_transform, train=True, download=args.download)
        test_set = torchvision.datasets.CIFAR10(args.datadir, transform=test_transform, train=False, download=args.download)
        train_indices = range(49000)
        valid_indices = range(49000, 50000)
        train_loader = torch.utils.data.DataLoader(train,
                                                batch_size=wandb.config['batch_size'],
                                                drop_last=True,
                                                sampler=torch.utils.data.SubsetRandomSampler(train_indices),
                                                num_workers=args.num_workers)
        valid_loader = torch.utils.data.DataLoader(valid,
                                                batch_size=wandb.config['batch_size'],
                                                drop_last=True,
                                                sampler=torch.utils.data.SubsetRandomSampler(valid_indices),
                                                num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
    else:
        train = torchvision.datasets.CIFAR10(args.datadir, transform=transform, train=True, download=args.download)
        valid = torchvision.datasets.CIFAR10(args.datadir, transform=test_transform, train=False, download=args.download)
        test_set = valid

        train_loader = torch.utils.data.DataLoader(train, batch_size=wandb.config['batch_size'],
                                                drop_last=True, num_workers=args.num_workers)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=wandb.config['batch_size'],
                                                drop_last=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)


    model = make_model()
    model.to(args.device)
    logger.info('model size: {}', sum(p.numel() for p in model.parameters()))

    criterion_ce = nn.CrossEntropyLoss()
    criterion_ls = nn.CrossEntropyLoss(label_smoothing=0.2)

    def criterion(y_hat, y):
        return 0.8 * criterion_ce(y_hat, y) + 0.2 * criterion_ls(y_hat, y)

    if True:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=wandb.config['learning_rate'],
                                    momentum=wandb.config['momentum'],
                                    weight_decay=wandb.config['weight_decay']) # type: ignore
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=wandb.config['learning_rate'],
                                    #momentum=wandb.config['momentum'],
                                    weight_decay=wandb.config['weight_decay']) # type: ignore
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=args.learning_rate,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)


    wandb.watch(model, log_graph=True)
    best_valid_accuracy = 0.
    checkpoint = str(Path(f'r9-{args.job_id}.pt'))

    for epoch in range(wandb.config['epochs']):
        reports = train_one_epoch(model, criterion, optimizer, train_loader, epoch=epoch, scheduler=scheduler, scaler=scaler)
        with torch.inference_mode():
            reports.update(evaluate(model, criterion, valid_loader, epoch=epoch))
            logger.info('{}', reports)
            if reports['valid/accuracy'] > best_valid_accuracy:
                torch.save({'model': model.state_dict(),
                            'reports': reports}, checkpoint)
                logger.info('new best: {}', checkpoint)
                best_valid_accuracy = reports['valid/accuracy']

    best_model = torch.load(checkpoint)
    logger.info('loading best model to test: {}', best_model['reports'])
    model.load_state_dict(best_model['model'])
    with torch.inference_mode():
        test_reports = test(model, test_loader)
        logger.info('test: {}', test_reports)

    wandb.save(checkpoint)

