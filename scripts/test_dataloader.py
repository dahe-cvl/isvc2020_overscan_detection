
import torchvision
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from datetime import datetime
import json
import argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from calculateStatistics import calculateMean, calculateSTD, saveStatistics
from models import *
from metrics import *
from utils import *


db_path = "/data/share/frame_border_detection_db_v6/rgb_3class_large/"

trainloader, validloader, testloader = loadSegDataset(data_dir=db_path,
                                          batch_size=8,
                                          expType="gray",
                                          data_aug_flag=True,
                                          dim=(360, 480))
print(len(testloader))


writer = SummaryWriter(log_dir="./runs/" + "dataloader_test")

for i, sample in enumerate(trainloader):
    inputs = sample[0]
    labels = sample[1]

    ## Convert torch tensor to Variable
    inputs = Variable(inputs)
    labels = Variable(labels)

    print(inputs.size())
    print(labels.size())
    #exit()

    tmp = np.expand_dims(labels.detach().cpu().numpy(), axis=1)
    print(tmp.shape)
    grid = torchvision.utils.make_grid(inputs.detach().cpu())
    writer.add_image("input", grid, 0)
    grid = torchvision.utils.make_grid(torch.Tensor(tmp))
    writer.add_image("masks", grid, 0)

    if (i >= 10):
        exit()

writer.close()