import time
import torch.optim

from adv_lib.mart import mart_loss_bd
from training import _jensen_shannon_div
from utils.utils import AverageMeter

import numpy as np
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(P, epoch, model, criterion, optimizer, scheduler, loader, adversary, logger=None):

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['mrt'] = AverageMeter()
    losses['con'] = AverageMeter()
    losses['adv'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        labels = labels.to(device)

        batch_size = images[0].size(0)
        images_aug1, images_aug2 = images[0].to(device), images[1].to(device)
        images_pair = torch.cat([images_aug1, images_aug2], dim=0)  # 2B

        loss_adv, loss_mart, outputs_adv, images_adv = mart_loss_bd(
            P, model, images_pair, labels.repeat(2), optimizer, distance=P.distance,
            eps_iter=P.alpha, eps=P.epsilon, nb_iter=P.n_iters,
            beta=P.beta, clip_min=0, clip_max=1, return_adv_sample=True
        )
        loss = loss_mart + loss_adv

        # divide non-boundary, boundary and misclassified samples
        model.eval()
        eval_output_benign = model(images_pair)
        eval_predicted_labels_benign = torch.argmax(eval_output_benign, 1).cpu().data.numpy()
        model.train()

        labels_np = deepcopy(labels.repeat(2)).cpu().numpy()
        all_index = np.array([i for i in range(len(labels_np))])
        misclassified_index = all_index[eval_predicted_labels_benign != labels_np]

        # apply BD regularization
        ori_train_len = int(len(labels_np) / 2)
        ori_all_index = [idx for idx in range(ori_train_len)]
        misclassified_index_1 = [idx for idx in misclassified_index if idx < ori_train_len]
        misclassified_index_2 = [idx - ori_train_len for idx in misclassified_index if idx >= ori_train_len]
        ori_misclassified_index_set = set(misclassified_index_1) | set(misclassified_index_2)
        ICT_index = np.sort(np.array(list(set(ori_all_index) - ori_misclassified_index_set)))

        if len(ICT_index) > 0:
            _lambda = np.random.beta(P.BD_alpha, P.BD_alpha)
            mixup_rate = max(_lambda, (1 - _lambda))
            images_adv1, images_adv2 = images_adv.chunk(2)
            ICT_data = (mixup_rate * images_adv1[ICT_index] + (1 - mixup_rate) * images_adv2[ICT_index])
            ICT_output_1 = model(ICT_data)

            outputs_benign = model(images_pair)
            outputs_benign1, outputs_benign2 = outputs_benign.chunk(2)
            ICT_output_2 = (mixup_rate * outputs_benign1[ICT_index] + (1 - mixup_rate) * outputs_benign2[ICT_index])

            loss_con = P.lam * _jensen_shannon_div(ICT_output_1, ICT_output_2, P.T)
            loss += loss_con

        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Log losses ###
        losses['mrt'].update(loss_mart.item(), batch_size)
        losses['adv'].update(loss_adv.item(), batch_size)
        if len(ICT_index) > 0:
            losses['con'].update(loss_con.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossMRT %f] [LossCon %f] [LossAdv %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['mrt'].value, losses['con'].value,
                  losses['adv'].value))

        check = time.time()

    if P.optimizer == 'sgd':
        scheduler.step()

    log_('[DONE] [Time %.3f] [Data %.3f] [LossMRT %f] '
         '[LossCon %f] [LossAdv %f]' %
         (batch_time.average, data_time.average,
          losses['mrt'].average, losses['con'].average,
          losses['adv'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_mart', losses['mrt'].average, epoch)
        logger.scalar_summary('train/loss_con', losses['con'].average, epoch)
        logger.scalar_summary('train/loss_adversary', losses['adv'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
