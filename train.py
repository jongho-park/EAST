import os
import os.path as osp
import time
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataset import EASTDataset
from model import EAST
from loss import Loss


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):
    trainset = EASTDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)
    file_num = len(trainset)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)

    criterion = Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model.train()
    for epoch in range(max_epoch):
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = (img.to(device), gt_score.to(device),
                                                  gt_geo.to(device), ignored_map.to(device))
            pred_score, pred_geo = model(img)
            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss '
                  'is {:.8f}'.format(epoch + 1, max_epoch, i + 1, int(file_num / batch_size),
                                     time.time() - start_time, loss.item()))

        scheduler.step()

        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(
            epoch_loss / int(file_num / batch_size), time.time() - epoch_time))
        print(time.asctime(time.localtime(time.time())))
        print('=' * 50)

        if (epoch + 1) % save_interval == 0:
            ckpt_fpath = osp.join(model_dir, 'model_epoch_{}.pth'.format(epoch + 1))
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
