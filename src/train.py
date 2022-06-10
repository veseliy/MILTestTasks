import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import time
import torch.backends.cudnn as cudnn
import torch.utils.data
from src.model_s import SSD300, MultiBoxLoss
from src.utils_s import *
from src.datasets import MILDataset

# Model training params

# Source: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = 3  # number of different types of objects
BATCH_SIZE = 8
num_workers = 2

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = BATCH_SIZE  # batch size
iterations = 120000  # number of iterations to train
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True

train_dataset = MILDataset('../dataset','train')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           collate_fn=train_dataset.collate_fn,
                                          num_workers=num_workers)

model = SSD300(n_classes=n_classes)
device = 'cuda'

criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

# if checkpoint is None:
#     start_epoch = 0
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SSD300(n_classes=n_classes)
    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
biases = list()
not_biases = list()
for param_name, param in model.named_parameters():
    if param.requires_grad:
        if param_name.endswith('.bias'):
            biases.append(param)
        else:
            not_biases.append(param)


optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                lr=lr, momentum=momentum, weight_decay=weight_decay)


model.train()  # training mode enables dropout

batch_time = AverageMeter()  # forward prop. + back prop. time
data_time = AverageMeter()  # data loading time
losses = AverageMeter()  # loss

start = time.time()
epoch = 1
for i, (images, boxes, labels) in enumerate(train_loader):
    #         # Move to default device
    images = images.to(device)  # (batch_size (N), 3, 300, 300)

    boxes = [b.to(device) for b in boxes]
    labels = [l.to(device) for l in labels]

    #         # Forward prop.
    predicted_locs, predicted_scores = model(images.float())  # (N, 8732, 4), (N, 8732, n_classes)

    loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#     losses.update(loss.item(), images.size(0))
#     batch_time.update(time.time() - start)

#     start = time.time()

#     # Print status
#     if i % print_freq == 0:
#         print('Epoch: [{0}][{1}/{2}]\t'
#               'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#               'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
#               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch,
#                                                               i,
#                                                               len(train_loader),
#                                                               batch_time=batch_time,
#                                                               data_time=data_time, loss=losses))
# del predicted_locs, predicted_scores, images, boxes, labels
