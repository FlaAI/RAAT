import time
import torch.optim
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
    losses['cls'] = AverageMeter()
    losses['con'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        batch_size = images[0].size(0)
        labels = labels.to(device)

        images_aug1, images_aug2 = images[0].to(device), images[1].to(device)
        images_pair = torch.cat([images_aug1, images_aug2], dim=0)  # 2B
        images_adv = adversary(images_pair, labels.repeat(2))

        # --- divide non-boundary, boundary and misclassified samples --- #
        model.eval()

        eval_output_benign = model(images_pair)
        eval_predicted_labels_benign = torch.argmax(eval_output_benign, 1).cpu().data.numpy()

        eval_output_adv = model(images_adv)
        eval_predicted_labels_adv = torch.argmax(eval_output_adv, 1).cpu().data.numpy()

        adv_perturb = images_adv.clone().detach() - images_pair.clone().detach()
        adv_input_reduced = images_pair.clone().detach() + P.BD_boundary_range * adv_perturb
        eval_output_adv_reduced = model(adv_input_reduced)
        eval_predicted_labels_adv_reduced = torch.argmax(eval_output_adv_reduced, 1).cpu().data.numpy()

        model.train()

        labels_np = deepcopy(labels.repeat(2)).cpu().numpy()
        all_index = np.array([i for i in range(len(labels_np))])
        misclassified_index = all_index[eval_predicted_labels_benign != labels_np]
        boundary_index = all_index[(eval_predicted_labels_benign == labels_np)
                                   & (eval_predicted_labels_adv != labels_np)
                                   & (eval_predicted_labels_adv_reduced != labels_np)]
        non_boundary_index = np.sort(
            np.array(list(set(list(all_index)) - set(list(misclassified_index)) - set(list(boundary_index)))))

        # reconstruct supervised set for CE loss
        data_ce = torch.cat((
            images_adv[non_boundary_index], adv_input_reduced[boundary_index], images_adv[misclassified_index]))
        labels_ce = torch.cat((
            labels.repeat(2)[non_boundary_index], labels.repeat(2)[boundary_index], labels.repeat(2)[misclassified_index]))

        outputs_ce = model(data_ce)
        loss_ce = criterion(outputs_ce, labels_ce)
        loss = loss_ce

        ### ICT regularization ###
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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Log losses ###
        losses['cls'].update(loss_ce.item(), batch_size)
        if len(ICT_index) > 0:
            losses['con'].update(loss_con.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossCon %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['con'].value))

        check = time.time()

    if P.optimizer == 'sgd':
        scheduler.step()

    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossCon %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average, losses['con'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/loss_con', losses['con'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
