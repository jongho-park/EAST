import os
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from dataset import EASTDataset
from model import EAST
from loss import Loss


def train(dataset_dir, pths_path, batch_size, lr, num_workers, epoch_iter, interval):
    trainset = EASTDataset(dataset_dir, split='train')
    file_num = len(trainset)
    train_loader = DataLoader(trainset, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers, drop_last=True)

    criterion = Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter // 2], gamma=0.1)

    model.train()
    for epoch in range(epoch_iter):
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
                  'is {:.8f}'.format(epoch + 1, epoch_iter, i + 1, int(file_num / batch_size),
                                     time.time() - start_time, loss.item()))

        scheduler.step()

        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss / int(file_num / batch_size), time.time() - epoch_time))
        print(time.asctime(time.localtime(time.time())))
        print('=' * 50)
        if (epoch + 1) % interval == 0:
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch + 1)))


if __name__ == '__main__':
    dataset_dir = '/data/datasets/ICDAR17_KoreanLatin'
    pths_path = './pths'
    batch_size = 12
    lr = 1e-3
    num_workers = 4
    epoch_iter = 600
    save_interval = 5
    train(dataset_dir, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval)
