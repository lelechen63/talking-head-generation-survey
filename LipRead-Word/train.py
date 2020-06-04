import os
import pdb
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils
from option import get_parser

class PadSequence:
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True)
        lengths = torch.LongTensor([len(x) for x in sequences])
        labels = [x[1] for x in sorted_batch]
        
        return sequences_padded, torch.LongTensor(labels), batch[2], lengths


def train(opt):
    utils.init_log_dir(opt)
    writer = SummaryWriter('./save/{}/tb'.format(opt.name))

    train_set, _, _ = utils.get_dataset(opt)

    train_loader = DataLoader(train_set, batch_size=opt.batch_size,
                              shuffle=True, num_workers=opt.num_workers, drop_last=True)

    encode, middle, decode = utils.get_model(opt)
    checkpoint = torch.load('./repo/lrs3_500_pretrain.pth.tar', map_location='cpu')
    encode.load_state_dict(checkpoint['encode'])
    if opt.gpu:
        encode, middle, decode = encode.cuda(), middle.cuda(), decode.cuda()
    if opt.gpus:
        encode, middle, decode = nn.DataParallel(
            encode), nn.DataParallel(middle), nn.DataParallel(decode)

    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam([{'params': encode.parameters(), 'lr': opt.lr * 0.1},
                            {'params': middle.parameters()},
                            {'params': decode.parameters()}], opt.lr, weight_decay=opt.weight_decay)

    scheduler = utils.AdjustLR(
        optimizer, [opt.lr * 0.1, opt.lr, opt.lr], sleep_epochs=1, half=5)

    for epoch in range(opt.epoches):
        scheduler.step(epoch)
        for step, pack in enumerate(train_loader):
            v = pack[0]
            align = pack[1]
            if opt.gpu:
                v = v.cuda()
                align = align.cuda()

            embeddings = encode(v)
            embeddings = middle(embeddings)

            digits = decode(embeddings)

            if opt.loss_smooth == 0:
                loss = loss_func(digits, align)
            else:
                smoothed_one_hot = utils.one_hot(opt, align, opt.out_channel)
                smoothed_one_hot = smoothed_one_hot * \
                    (1 - opt.loss_smooth) + (1 - smoothed_one_hot) * \
                    opt.loss_smooth / (opt.out_channel - 1)

                log_prb = F.log_softmax(digits, dim=1)
                loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1) % 10 == 0:
                res = torch.argmax(digits.detach(), -1).cpu().numpy()
                top_1_cls = np.mean(res == align.cpu().numpy())
                print('epoch:{} step:{}/{} train_loss:{:.4f} train_acc:{:.4f}'.format(
                    epoch, (step+1), len(train_loader), loss.item(), top_1_cls))
                writer.add_scalar('train-loss', loss.item(), epoch * len(train_loader) + step + 1)
                writer.add_scalar('train-top1', top_1_cls, epoch * len(train_loader) + step + 1)

        torch.save({'encode': encode.module.state_dict() if opt.gpus else encode.state_dict(),
                    'mid_net': middle.module.state_dict() if opt.gpus else middle.state_dict(),
                    'decode': decode.module.state_dict() if opt.gpus else decode.state_dict(),
                    'optimizer': optimizer}, os.path.join('./save', opt.name, 'check', 'model_{}_{}.pth.tar'.format(epoch, step+1)))


def test(opt):
    _, _, test_set = utils.get_dataset(opt)

    test_loader = DataLoader(test_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=opt.num_workers, drop_last=False)

    encode, middle, decode = utils.get_model(opt)
    checkpoint = torch.load(opt.load, map_location='cpu')
    encode.load_state_dict(checkpoint['encode'])
    middle.load_state_dict(checkpoint['mid_net'])
    decode.load_state_dict(checkpoint['decode'])
    encode, middle, decode = encode.eval(), middle.eval(), decode.eval()
    if opt.gpu:
        encode, middle, decode = encode.cuda(), middle.cuda(), decode.cuda()

    assert not opt.gpus

    res = []
    gt = []
    with torch.no_grad():
        for step, pack in enumerate(tqdm(test_loader)):
            v = pack[0]
            align = pack[1]
            if opt.gpu:
                v = v.cuda()

            v = v.transpose(1, 2)
            v = v.reshape([-1]+list(v.shape[2:]))
            embeddings = encode(v)
            embeddings = middle(embeddings)

            digits = decode(embeddings).reshape([-1, 10]+list(embeddings.shape[1:])).mean(1).cpu()
            res.append(digits)
            gt.append(align)

        res = torch.cat(res, 0)
        gt = torch.cat(gt, 0)
        top1, top5, top10 = utils.accuracy(res, gt, (1, 5, 10))

    assert opt.topw <= len(test_set.labels)
    if opt.topw == len(test_set.labels):
        print('test top1:{}'.format(top1))
        print('test top5:{}'.format(top5))
        print('test top10:{}'.format(top10))
    else:
        cat_dict = {}
        for l in list(set(gt)):
            if len(gt[gt==l])>0:
                top1, top5, top10 = utils.accuracy(res[gt==l], gt[gt==l], (1, 5, 10))
                cat_dict[l] = top1
            else:
                cat_dict[l] = 0
        cat_dict = {k: v.item() for k, v in sorted(cat_dict.items(), key=lambda item: item[1], reverse=True)}

        gt_idx = [True if g in list(cat_dict.keys())[:opt.topw] else False for g in gt]
        top1, top5, top10 = utils.accuracy(res[gt_idx], gt[gt_idx], (1, 5, 10))
        print('test top1:{}'.format(top1))
        print('test top5:{}'.format(top5))
        print('test top10:{}'.format(top10))


if __name__ == '__main__':
    opt = get_parser()
    # train(opt)
    test(opt)