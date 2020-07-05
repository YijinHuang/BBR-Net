import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

from metrics import accuracy, quadratic_weighted_kappa, bbox_iou, bbox_giou

import os
import cv2 as cv
from tqdm import tqdm

from PIL import Image


def train(model, train_dataset, val_dataset, epochs, learning_rate, batch_size, save_path):
    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    # define loss and optimizier
    # cross_entropy = nn.MSELoss()
    def GIouLoss(pred, target):
        giou = bbox_giou(pred, target).mean()
        return 1 - giou

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)

    # learning rate warmup and decay
    warmup_epoch = 10
    warmup_batch = len(train_loader) * warmup_epoch
    remain_batch = len(train_loader) * (epochs - warmup_epoch)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remain_batch)
    warmup_scheduler = WarmupLRScheduler(optimizer, warmup_batch, learning_rate)

    # train
    record_epochs, accs, losses = _train(
        model,
        train_loader,
        val_loader,
        GIouLoss,
        optimizer,
        epochs,
        save_path,
        lr_scheduler=lr_scheduler,
        warmup_scheduler=warmup_scheduler,
    )
    return model, record_epochs, accs, losses


def _train(model, train_loader, val_loader, loss_function, optimizer, epochs, save_path,
           weighted_sampler=None, lr_scheduler=None, extra_loss=None, warmup_scheduler=None):
    model_dict = model.state_dict()
    trainable_layers = [(tensor, model_dict[tensor].size()) for tensor in model_dict]
    print_msg('Trainable layers: ', ['{}\t{}'.format(k, v) for k, v in trainable_layers])

    # train
    max_acc = 0
    record_epochs, accs, losses = [], [], []
    model.train()
    for epoch in range(1, epochs + 1):
        # resampling weight update
        if weighted_sampler:
            weighted_sampler.step()

        # learning rate update
        if warmup_scheduler and not warmup_scheduler.is_finish():
            if epoch > 1:
                curr_lr = optimizer.param_groups[0]['lr']
                print_msg('Learning rate warmup to {}'.format(curr_lr))
        elif lr_scheduler:
            if epoch % 10 == 0:
                curr_lr = optimizer.param_groups[0]['lr']
                print_msg('Current learning rate is {}'.format(curr_lr))

        total = 0
        correct = 0
        epoch_loss = 0
        progress = tqdm(enumerate(train_loader))
        for step, train_data in progress:
            if warmup_scheduler and not warmup_scheduler.is_finish():
                warmup_scheduler.step()
            elif lr_scheduler:
                lr_scheduler.step()

            X, y = train_data
            X, y = X.cuda(), y.float().cuda()

            # forward
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            if extra_loss:
                extra_loss_value = extra_loss(model, X, y_pred, y)
                loss += extra_loss_value

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            epoch_loss += loss.item()
            total += y.size(0)
            correct += accuracy(y_pred, y) * y.size(0)
            avg_loss = epoch_loss / (step + 1)
            avg_acc = correct / total
            progress.set_description(
                'epoch: {}, loss: {:.6f}, acc: {:.4f}'
                .format(epoch, avg_loss, avg_acc)
            )

        # save model
        acc = _eval(model, val_loader)
        print('validation accuracy: {}'.format(acc))
        if acc > max_acc:
            torch.save(model, save_path)
            max_acc = acc
            print_msg('Model save at {}'.format(save_path))

        # record
        record_epochs.append(epoch)
        accs.append(acc)
        losses.append(avg_loss)

    return record_epochs, accs, losses


def evaluate(model_path, test_dataset, save_path):
    trained_model = torch.load(model_path).cuda()
    test_acc = visualize(trained_model, test_dataset, save_path)
    print('========================================')
    print('Finished! test acc: {}'.format(test_acc))
    print('========================================')


def _eval(model, dataloader, c_matrix=None):
    model.eval()
    torch.set_grad_enabled(False)

    correct = 0
    total = 0
    for step, test_data in enumerate(dataloader):
        X, y = test_data
        X, y = X.cuda(), y.float().cuda()

        y_pred = model(X)
        total += y.size(0)
        correct += accuracy(y_pred, y, c_matrix) * y.size(0)
    acc = round(correct.item() / total, 4)

    model.train()
    torch.set_grad_enabled(True)
    return acc


def visualize(model, dataset, save_path):
    model.eval()
    torch.set_grad_enabled(False)

    avg_iou = 0
    for index in tqdm(range(len(dataset))):
        X, y = dataset[index]
        X, y = X.cuda().unsqueeze(0), y

        y_pred = model(X)

        iou = bbox_iou(y_pred, torch.from_numpy(y).unsqueeze(0).float().cuda())[0].item()
        avg_iou += iou

        import metrics
        x1, y1, x2, y2 = metrics.to_corner(y[0], y[1], y[2], y[3])
        # pred_x1, pred_y1, pred_x2, pred_y2 = y_pred.cpu().numpy().tolist()[0]
        y_pred = y_pred[0]
        pred_x1, pred_y1, pred_x2, pred_y2 = metrics.to_corner(y_pred[0], y_pred[1], y_pred[2], y_pred[3])


        path = dataset.image_names[index]
        img = cv.imread(path)
        origin_path = path
        # origin_path = path.replace('infer_gaoyao_cropped_img_train', 'origin_gaoyao_cropped_img_train')
        origin_img = cv.imread(origin_path)
        origin_img = padding_resize(origin_img)
        orgin_img_shape = origin_img.shape
        img_shape = img.shape

        x_offset = int((orgin_img_shape[1] - img_shape[1]) / 2)
        y_offset = int((orgin_img_shape[0] - img_shape[0]) / 2)

        cv.rectangle(
            origin_img,
            (int(x1*224), int(y1*224)), 
            (int(x2*224), int(y2*224)), 
            (0, 255, 0), 
            1
        )
        # cv.rectangle(
        #     origin_img, 
        #     (int(pred_x1*img_shape[1] + x_offset), int(pred_y1*img_shape[0]) + y_offset), 
        #     (int(pred_x2*img_shape[1] + x_offset), int(pred_y2*img_shape[0]) + y_offset),
        #     (255, 0, 0),
        #     1
        # )
        cv.rectangle(
            origin_img, 
            (int(pred_x1*224), int(pred_y1*224)), 
            (int(pred_x2*224), int(pred_y2*224)),
            (255, 0, 0),
            1
        )

        img_name = os.path.split(path)[-1]
        new_name = '{}_{}.jpg'.format(os.path.splitext(img_name)[0], iou)
        new_path = os.path.join(save_path, new_name)
        cv.imwrite(new_path, origin_img)
    
    print(avg_iou)
    print(avg_iou / len(dataset))


def padding_resize(img):
    desired_size = 224

    img = Image.fromarray(img, mode='RGB')
    old_size = img.size

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    img = img.resize(new_size, Image.ANTIALIAS)

    new_img = Image.new("RGB", (desired_size, desired_size))
    new_img.paste(img, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

    return np.array(new_img)


def print_msg(msg, appendixs=[]):
    max_len = len(max([msg, *appendixs], key=len))
    print('=' * max_len)
    print(msg)
    for appendix in appendixs:
        print(appendix)
    print('=' * max_len)


# reference: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class WarmupLRScheduler:
    def __init__(self, optimizer, warmup_batch, initial_lr):
        self.step_num = 1
        self.optimizer = optimizer
        self.warmup_batch = warmup_batch
        self.initial_lr = initial_lr

    def step(self):
        if self.step_num <= self.warmup_batch:
            self.step_num += 1
            curr_lr = (self.step_num / self.warmup_batch) * self.initial_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = curr_lr

    def is_finish(self):
        return self.step_num > self.warmup_batch
