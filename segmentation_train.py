import torchvision
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import json
import argparse
from datetime import datetime
import pathlib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from scripts.calculateStatistics import calculateMean, calculateSTD, saveStatistics
from models import *
from metrics import *
from utils import *


def calculateMetrics(mask_gt, mask_pred, n_classes, tOverall_acc_sum, tAvg_per_class_acc_sum, tAvg_jacc_sum, tAvg_dice_sum):
    #################################
    ## evaluate
    #################################
    mask_pred = mask_pred.detach().cpu()
    mask_gt = mask_gt.detach().cpu()
    overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(mask_gt, mask_pred, n_classes)
    tOverall_acc_sum += overall_acc
    tAvg_per_class_acc_sum += avg_per_class_acc
    tAvg_jacc_sum += avg_jacc
    tAvg_dice_sum += avg_dice
    return tOverall_acc_sum, tAvg_per_class_acc_sum, tAvg_jacc_sum, tAvg_dice_sum

def csvLogger(dst_folder="", name="metrics_history.log", expName="nan", epoch=-1, entries_list=None):
    if(epoch == -1):
        print("ERROR: epoch must have a valid entry!")
    if (entries_list == None):
        print("ERROR: entries_list must have a valid entry!")

    # prepare entry_line
    entry_line = str(expName) + ";" + str(epoch) + ";"
    for i in range(0, len(entries_list)):
        tmp = entries_list[i]
        entry_line = entry_line + str(tmp) + ";"

    fp = open(dst_folder + "/" + str(name), 'a')
    fp.write(entry_line + "\n")
    fp.close()

#######################
## train model
#######################

def train(param_dict):
    if(param_dict == None):
        print("ERROR: you have to specify a valid json config!")
        exit()

    dst_path = param_dict['dst_path']
    expTimeStamp = datetime.now().strftime("%Y%m%d_%H%M")
    expType = param_dict['expType']
    expNet = param_dict['expNet']
    pre_trained_weights = param_dict['pre_trained_weights']
    expNum = param_dict['expNum']
    db_path = param_dict['db_path']
    n_epochs = param_dict['n_epochs']
    batch_size = param_dict['batch_size']
    lRate = param_dict['lRate']
    wDecay = param_dict['wDecay']
    classes = param_dict['classes']
    early_stopping_threshold = param_dict['early_stopping_threshold']
    loss_metric = param_dict['loss_metric']
    data_aug_flag = param_dict['data_aug_flag']
    dim = param_dict['resized_dim']
    activate_lower_features = bool(param_dict['activate_lower_features'])
    trainable_backbone_flag = bool(param_dict['trainable_backbone_flag'])

    CSV_LOGGER_FLAG = True
    TENSORBOARD_LOGGER_FLAG = True
    SAVE_CHECKPOINTS_FLAG = True
    TIME_LOG = True

    # create dst_folder
    createFolder(dst_path)

    ####################
    ## create experiment
    ####################
    expName = str(expTimeStamp) + "_" + str(expType) + "_" + str(expNet) + "_ExpNum_" + str(expNum)
    expFolder = dst_path + "/" + expName

    if not os.path.isdir(dst_path + "/" + expName):
        os.mkdir(expFolder)

    with open(expFolder + "/experiment_notes.json", 'w') as json_file:
        json.dump(param_dict, json_file)

    if (TENSORBOARD_LOGGER_FLAG == True):
        writer = SummaryWriter(log_dir="./runs/" + expName)

    ################
    # load dataset
    ################
    trainloader, validloader, testloader = loadSegDataset(data_dir=db_path,
                                                          batch_size=batch_size,
                                                          expType=expType,
                                                          data_aug_flag=data_aug_flag,
                                                          dim=dim)

    train_on_gpu = torch.cuda.is_available()
    print("Train on gpu: " + str(train_on_gpu))

    # Number of gpus
    multi_gpu = False;
    if train_on_gpu:
        gpu_count = torch.cuda.device_count()
        print("gpu_count: " + str(gpu_count))
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    ################
    # define model
    ################

    if (loss_metric == "bce_loss"):
        classes = ["1"]
        model = loadModel(model_arch=expNet,
                          classes=classes,
                          pre_trained_path=pre_trained_weights,
                          expType=expType,
                          lower_features=activate_lower_features)
    else:
        model, features = loadModel(model_arch=expNet,
                                    classes=classes,
                                    pre_trained_path=pre_trained_weights,
                                    expType=expType,
                                    lower_features=activate_lower_features,
                                    trainable_backbone_flag=trainable_backbone_flag)
    print("loaded successfully")

    if train_on_gpu:
        model = model.to('cuda')

    if multi_gpu:
        model = nn.DataParallel(model)

    ################
    # Specify the Loss function
    # https://pythonawesome.com/semantic-segmentation-models-datasets-and-losses-implemented-in-pytorch/
    ################
    if(loss_metric == "mse"):
        criterion = nn.MSELoss()
    elif (loss_metric == "cross_entropy"):
        criterion = nn.CrossEntropyLoss()
    elif (loss_metric == "focal_loss"):
        criterion = FocalLoss()
    elif (loss_metric == "bce_loss"):
        criterion = nn.BCEWithLogitsLoss()

    ################
    # Specify the optimizer
    ################
    optimizer = optim.SGD(model.parameters(), lr=lRate, momentum=0.9, nesterov=True, weight_decay=wDecay)
    #ptimizer = optim.Adam(model.parameters(), lr=lRate, weight_decay=wDecay)

    print("[Creating Learning rate scheduler...]")
    #steps = 10
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # Define the lists to store the results of loss and accuracy
    best_acc = 0.0
    best_loss = 10.0
    early_stopping_cnt = 0


    for epoch in range(0, n_epochs):
        if(TIME_LOG == True):
            start = datetime.now()

        tLoss_sum = 0
        tAcc_sum = 0
        vLoss_sum = 0
        vAcc_sum = 0
        tOverall_acc_sum = 0
        tAvg_per_class_acc_sum = 0
        tAvg_jacc_sum = 0
        tAvg_dice_sum = 0
        vOverall_acc_sum = 0
        vAvg_per_class_acc_sum = 0
        vAvg_jacc_sum = 0
        vAvg_dice_sum = 0
        ###################
        # train the model #
        ###################
        model.train()
        for i, sample in enumerate(trainloader):
            #print("train: " + str(i))

            inputs = sample[0]
            labels = sample[1]
            labels = labels.long()

            # Convert torch tensor to Variable
            inputs = Variable(inputs)
            labels = Variable(labels)

            # If we have GPU, shift the data to GPU
            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            #print("DEBUG A")
            #f = features(inputs)
            #print(f.size())
            #exit()

            outputs = model(inputs)['out']
            tLoss = criterion(outputs, labels)
            tLoss_sum += tLoss.item()

            if (len(classes) == 2):
                outputs = torch.sigmoid(outputs)

            outputs = torch.softmax(outputs, dim=1)
            mask_pred = outputs.argmax(1, keepdim=True)
            mask_pred = torch.squeeze(mask_pred)

            #print(inputs.size())
            #print(labels.size())
            #print(outputs.size())
            #print(mask_pred.size())
            # print(labels.dtype)
            #exit()

            # run backward pass
            optimizer.zero_grad()
            tLoss.backward()
            optimizer.step()

            #################################
            ## evaluate
            #################################

            tOverall_acc_sum, tAvg_per_class_acc_sum, tAvg_jacc_sum, tAvg_dice_sum = calculateMetrics(labels, mask_pred,
                                                                                                     len(classes),
                                                                                                     tOverall_acc_sum,
                                                                                                     tAvg_per_class_acc_sum,
                                                                                                     tAvg_jacc_sum,
                                                                                                     tAvg_dice_sum)

        ''''''
        ###################
        # validate the model #
        ###################
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, sample in enumerate(validloader):
                #print("val: " + str(i))
                inputs = sample[0]
                labels = sample[1]
                labels = labels.long()

                ## Convert torch tensor to Variable
                inputs = Variable(inputs)
                labels = Variable(labels)

                # If we have GPU, shift the data to GPU
                CUDA = torch.cuda.is_available()
                if CUDA:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = model(inputs)['out']
                vLoss = criterion(outputs, labels)
                vLoss_sum += vLoss.item()

                if (len(classes) == 2):
                    outputs = torch.sigmoid(outputs)

                outputs = torch.softmax(outputs, dim=1)
                mask_pred = outputs.argmax(1, keepdim=True)
                mask_pred = torch.squeeze(mask_pred)

                #################################
                ## evaluate
                #################################
                vOverall_acc_sum, vAvg_per_class_acc_sum, vAvg_jacc_sum, vAvg_dice_sum = calculateMetrics(labels,
                                                                                                          mask_pred,
                                                                                                          len(classes),
                                                                                                          vOverall_acc_sum,
                                                                                                          vAvg_per_class_acc_sum,
                                                                                                          vAvg_jacc_sum,
                                                                                                          vAvg_dice_sum)

                #grid = torchvision.utils.make_grid(inputs)
                #writer.add_image("input", grid, 0)
                #grid = torchvision.utils.make_grid(labels)
                #writer.add_image("masks", grid, 0)
                #grid = torchvision.utils.make_grid(preds)
                #writer.add_image("output", grid, 0)

        print('Epoch [{:d}/{:d}]: train_loss: {:.6f}, val_loss: {:.6f}'.format(epoch + 1,
                                                                               n_epochs,
                                                                               tLoss_sum / len(trainloader),
                                                                               vLoss_sum / len(validloader)))

        if (TIME_LOG == True):
            stop = datetime.now()

        if (TIME_LOG == True):
            time_diff = stop - start
            print("time_diff per epoch: " + str(time_diff))

        ###############################
        # write results to tensorboard
        ###############################
        if (TENSORBOARD_LOGGER_FLAG == True):
            writer.add_scalar('train_loss', tLoss_sum / len(trainloader), epoch)
            writer.add_scalar('valid_loss', vLoss_sum / len(validloader), epoch)
            writer.add_scalar('train_overall_acc', tOverall_acc_sum / len(trainloader), epoch)
            writer.add_scalar('train_per_class_acc', tAvg_per_class_acc_sum / len(trainloader), epoch)
            writer.add_scalar('train_Avg_jacc', tAvg_jacc_sum / len(trainloader), epoch)
            writer.add_scalar('train_Avg_dice', tAvg_dice_sum / len(trainloader), epoch)
            writer.add_scalar('valid_overall_acc', vOverall_acc_sum / len(validloader), epoch)
            writer.add_scalar('valid_per_class_acc', vAvg_per_class_acc_sum / len(validloader), epoch)
            writer.add_scalar('valid_Avg_jacc', vAvg_jacc_sum / len(validloader), epoch)
            writer.add_scalar('valid_Avg_dice', vAvg_dice_sum / len(validloader), epoch)

        ###############################
        # write results to csv
        ###############################
        if(CSV_LOGGER_FLAG == True):
            entries_list = []
            entries_list.append(float(tLoss_sum / len(trainloader)))
            entries_list.append(float(vLoss_sum / len(validloader)))
            entries_list.append(float(tOverall_acc_sum / len(trainloader)))
            entries_list.append(float(vOverall_acc_sum / len(validloader)))
            entries_list.append(float(tAvg_per_class_acc_sum / len(trainloader)))
            entries_list.append(float(vAvg_per_class_acc_sum / len(validloader)))
            entries_list.append(float(tAvg_jacc_sum / len(trainloader)))
            entries_list.append(float(vAvg_jacc_sum / len(validloader)))
            entries_list.append(float(tAvg_dice_sum / len(trainloader)))
            entries_list.append(float(vAvg_dice_sum / len(validloader)))

            csvLogger(dst_folder=expFolder,
                      name="metrics_history.log",
                      epoch=epoch, entries_list=entries_list);

        ###############################
        # Save checkpoint.
        ###############################
        vloss_curr = vLoss_sum / len(validloader)
        tloss_curr = tLoss_sum / len(trainloader)

        if (SAVE_CHECKPOINTS_FLAG == True):
            #acc_curr = 100. * (vAcc_sum / len(validloader));
            if vloss_curr < best_loss:
                print('Saving...')
                state = {
                    'net': model.state_dict(),
                    'acc': None,
                    'loss': vloss_curr,
                    'epoch': epoch,
                }
                # if not os.path.isdir('checkpoint'):
                #    os.mkdir('checkpoint')
                torch.save(state,
                           expFolder + "/" "best_model" + ".pth")  # + str(round(acc_curr, 4)) + "_" + str(round(vloss_curr, 4))
                #best_acc = acc_curr
                best_loss = vloss_curr
                early_stopping_cnt = 0;
        scheduler.step(vloss_curr)

        ###############################
        # early stopping.
        ###############################
        if (vloss_curr >= best_loss):
            early_stopping_cnt = early_stopping_cnt + 1;
        if (early_stopping_cnt >= early_stopping_threshold):
            print('Early stopping active --> stop training ...')
            break;
        ''''''
    if (TENSORBOARD_LOGGER_FLAG == True):
        writer.close()

def trainThPredictor(param_dict):
    if(param_dict == None):
        print("ERROR: you have to specify a valid json config!")
        exit()

    dst_path = param_dict['dst_path']
    expTimeStamp = datetime.now().strftime("%Y%m%d_%H%M")
    expType = param_dict['expType']
    expNet = param_dict['expNet']
    pre_trained_weights = param_dict['pre_trained_weights']
    expNum = param_dict['expNum']
    db_path = param_dict['db_path']
    n_epochs = param_dict['n_epochs']
    batch_size = param_dict['batch_size']
    lRate = param_dict['lRate']
    wDecay = param_dict['wDecay']
    classes = param_dict['classes']
    early_stopping_threshold = param_dict['early_stopping_threshold']
    loss_metric = param_dict['loss_metric']
    data_aug_flag = param_dict['data_aug_flag']
    dim = param_dict['resized_dim']
    activate_lower_features = bool(param_dict['activate_lower_features'])
    trainable_backbone_flag = bool(param_dict['trainable_backbone_flag'])

    CSV_LOGGER_FLAG = True
    TENSORBOARD_LOGGER_FLAG = True
    SAVE_CHECKPOINTS_FLAG = True

    # create dst_folder
    createFolder(dst_path)

    ####################
    ## create experiment
    ####################
    expName = str(expTimeStamp) + "_" + str(expType) + "_" + str(expNet) + "_ExpNum_" + str(expNum)
    expFolder = dst_path + "/" + expName

    if not os.path.isdir(dst_path + "/" + expName):
        os.mkdir(expFolder)

    with open(expFolder + "/experiment_notes.json", 'w') as json_file:
        json.dump(param_dict, json_file)

    if (TENSORBOARD_LOGGER_FLAG == True):
        writer = SummaryWriter(log_dir="./runs/" + expName)

    ################
    # load dataset
    ################
    trainloader, validloader, testloader = loadSegDataset(data_dir=db_path,
                                                          batch_size=batch_size,
                                                          expType=expType,
                                                          data_aug_flag=data_aug_flag,
                                                          dim=dim)

    train_on_gpu = torch.cuda.is_available()
    print("Train on gpu: " + str(train_on_gpu))

    # Number of gpus
    multi_gpu = False;
    if train_on_gpu:
        gpu_count = torch.cuda.device_count()
        print("gpu_count: " + str(gpu_count))
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    ################
    # define model
    ################

    if (loss_metric == "bce_loss"):
        classes = ["1"]
        model = loadModel(model_arch=expNet,
                          classes=classes,
                          pre_trained_path=pre_trained_weights,
                          expType=expType,
                          lower_features=activate_lower_features)
    else:
        model, features = loadModel(model_arch=expNet,
                                    classes=classes,
                                    pre_trained_path=pre_trained_weights,
                                    expType=expType,
                                    lower_features=activate_lower_features,
                                    trainable_backbone_flag=trainable_backbone_flag)
    print("loaded successfully")

    if train_on_gpu:
        model = model.to('cuda')

    if multi_gpu:
        model = nn.DataParallel(model)

    ################
    # Specify the Loss function
    # https://pythonawesome.com/semantic-segmentation-models-datasets-and-losses-implemented-in-pytorch/
    ################
    if(loss_metric == "mse"):
        criterion = nn.MSELoss()
    elif (loss_metric == "cross_entropy"):
        criterion = nn.CrossEntropyLoss()
    elif (loss_metric == "focal_loss"):
        criterion = FocalLoss()
    elif (loss_metric == "bce_loss"):
        criterion = nn.BCEWithLogitsLoss()

    ################
    # Specify the optimizer
    ################
    optimizer = optim.SGD(model.parameters(), lr=lRate, momentum=0.9, nesterov=True, weight_decay=wDecay)
    #ptimizer = optim.Adam(model.parameters(), lr=lRate, weight_decay=wDecay)

    print("[Creating Learning rate scheduler...]")
    #steps = 10
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # Define the lists to store the results of loss and accuracy
    best_acc = 0.0
    best_loss = 10.0
    early_stopping_cnt = 0

    for epoch in range(0, n_epochs):
        tLoss_sum = 0
        tAcc_sum = 0
        vLoss_sum = 0
        vAcc_sum = 0
        tOverall_acc_sum = 0
        tAvg_per_class_acc_sum = 0
        tAvg_jacc_sum = 0
        tAvg_dice_sum = 0
        vOverall_acc_sum = 0
        vAvg_per_class_acc_sum = 0
        vAvg_jacc_sum = 0
        vAvg_dice_sum = 0
        ###################
        # train the model #
        ###################
        model.train()
        for i, sample in enumerate(trainloader):
            #print("train: " + str(i))

            inputs = sample[0]
            labels = sample[1]
            labels = labels.long()

            # Convert torch tensor to Variable
            inputs = Variable(inputs)
            labels = Variable(labels)

            # If we have GPU, shift the data to GPU
            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            #print("DEBUG A")
            #f = features(inputs)
            #print(f.size())
            #exit()

            thresholds = model(inputs)['out']
            print(thresholds)
            #print(thresholds.size())
            #print(inputs.size())

            outputs = inputs

            for t in range(0, batch_size):
                outputs[t][inputs[t] > thresholds[t]] = 1
                outputs[t][inputs[t] <= thresholds[t]] = 0
            #outputs[:, inputs <= thresholds] = 0
            outputs = outputs[:, 1, :]
            #print(outputs.size())
            #print(torch.unique(outputs))

            tLoss = criterion(outputs, labels)
            tLoss_sum += tLoss.item()
            #outputs = torch.softmax(outputs, dim=1)
            #mask_pred = outputs.argmax(1, keepdim=True)
            #mask_pred = torch.squeeze(mask_pred)

            #print(inputs.size())
            #print(labels.size())
            #print(outputs.size())
            #print(mask_pred.size())
            # print(labels.dtype)
            #exit()

            # run backward pass
            optimizer.zero_grad()
            tLoss.backward()
            optimizer.step()

            #################################
            ## evaluate
            #################################

            #tOverall_acc_sum, tAvg_per_class_acc_sum, tAvg_jacc_sum, tAvg_dice_sum = calculateMetrics(labels, mask_pred,
            #                                                                                         len(classes),
            #                                                                                         tOverall_acc_sum,
            ##                                                                                        tAvg_per_class_acc_sum,
            #                                                                                        tAvg_jacc_sum,
            #                                                                                        tAvg_dice_sum)

        print('Epoch [{:d}/{:d}]: train_loss: {:.6f}, val_loss: {:.6f}'.format(epoch + 1,
                                                                               n_epochs,
                                                                               tLoss_sum / len(trainloader),
                                                                               tLoss_sum / len(trainloader)))
        continue
        ''''''
        ###################
        # validate the model #
        ###################
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, sample in enumerate(validloader):
                #print("val: " + str(i))
                inputs = sample[0]
                labels = sample[1]
                labels = labels.long()

                ## Convert torch tensor to Variable
                inputs = Variable(inputs)
                labels = Variable(labels)

                # If we have GPU, shift the data to GPU
                CUDA = torch.cuda.is_available()
                if CUDA:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = model(inputs)['out']
                vLoss = criterion(outputs, labels)
                vLoss_sum += vLoss.item()

                if (len(classes) == 2):
                    outputs = torch.sigmoid(outputs)

                outputs = torch.softmax(outputs, dim=1)
                mask_pred = outputs.argmax(1, keepdim=True)
                mask_pred = torch.squeeze(mask_pred)

                #################################
                ## evaluate
                #################################
                vOverall_acc_sum, vAvg_per_class_acc_sum, vAvg_jacc_sum, vAvg_dice_sum = calculateMetrics(labels,
                                                                                                          mask_pred,
                                                                                                          len(classes),
                                                                                                          vOverall_acc_sum,
                                                                                                          vAvg_per_class_acc_sum,
                                                                                                          vAvg_jacc_sum,
                                                                                                          vAvg_dice_sum)

                #grid = torchvision.utils.make_grid(inputs)
                #writer.add_image("input", grid, 0)
                #grid = torchvision.utils.make_grid(labels)
                #writer.add_image("masks", grid, 0)
                #grid = torchvision.utils.make_grid(preds)
                #writer.add_image("output", grid, 0)

        print('Epoch [{:d}/{:d}]: train_loss: {:.6f}, val_loss: {:.6f}'.format(epoch + 1,
                                                                               n_epochs,
                                                                               tLoss_sum / len(trainloader),
                                                                               vLoss_sum / len(validloader)))

        ###############################
        # write results to tensorboard
        ###############################
        if (TENSORBOARD_LOGGER_FLAG == True):
            writer.add_scalar('train_loss', tLoss_sum / len(trainloader), epoch)
            writer.add_scalar('valid_loss', vLoss_sum / len(validloader), epoch)
            writer.add_scalar('train_overall_acc', tOverall_acc_sum / len(trainloader), epoch)
            writer.add_scalar('train_per_class_acc', tAvg_per_class_acc_sum / len(trainloader), epoch)
            writer.add_scalar('train_Avg_jacc', tAvg_jacc_sum / len(trainloader), epoch)
            writer.add_scalar('train_Avg_dice', tAvg_dice_sum / len(trainloader), epoch)
            writer.add_scalar('valid_overall_acc', vOverall_acc_sum / len(validloader), epoch)
            writer.add_scalar('valid_per_class_acc', vAvg_per_class_acc_sum / len(validloader), epoch)
            writer.add_scalar('valid_Avg_jacc', vAvg_jacc_sum / len(validloader), epoch)
            writer.add_scalar('valid_Avg_dice', vAvg_dice_sum / len(validloader), epoch)

        ###############################
        # write results to csv
        ###############################
        if(CSV_LOGGER_FLAG == True):
            entries_list = []
            entries_list.append(float(tLoss_sum / len(trainloader)))
            entries_list.append(float(vLoss_sum / len(validloader)))
            entries_list.append(float(tOverall_acc_sum / len(trainloader)))
            entries_list.append(float(vOverall_acc_sum / len(validloader)))
            entries_list.append(float(tAvg_per_class_acc_sum / len(trainloader)))
            entries_list.append(float(vAvg_per_class_acc_sum / len(validloader)))
            entries_list.append(float(tAvg_jacc_sum / len(trainloader)))
            entries_list.append(float(vAvg_jacc_sum / len(validloader)))
            entries_list.append(float(tAvg_dice_sum / len(trainloader)))
            entries_list.append(float(vAvg_dice_sum / len(validloader)))

            csvLogger(dst_folder=expFolder,
                      name="metrics_history.log",
                      epoch=epoch, entries_list=entries_list);

        ###############################
        # Save checkpoint.
        ###############################
        vloss_curr = vLoss_sum / len(validloader)
        tloss_curr = tLoss_sum / len(trainloader)

        if (SAVE_CHECKPOINTS_FLAG == True):
            #acc_curr = 100. * (vAcc_sum / len(validloader));
            if vloss_curr < best_loss:
                print('Saving...')
                state = {
                    'net': model.state_dict(),
                    'acc': None,
                    'loss': vloss_curr,
                    'epoch': epoch,
                }
                # if not os.path.isdir('checkpoint'):
                #    os.mkdir('checkpoint')
                torch.save(state,
                           expFolder + "/" "best_model" + ".pth")  # + str(round(acc_curr, 4)) + "_" + str(round(vloss_curr, 4))
                #best_acc = acc_curr
                best_loss = vloss_curr
                early_stopping_cnt = 0;
        scheduler.step(vloss_curr)

        ###############################
        # early stopping.
        ###############################
        if (vloss_curr >= best_loss):
            early_stopping_cnt = early_stopping_cnt + 1;
        if (early_stopping_cnt >= early_stopping_threshold):
            print('Early stopping active --> stop training ...')
            break;
        ''''''
    if (TENSORBOARD_LOGGER_FLAG == True):
        writer.close()

def trainSingleExpThPredictor(config_dict=None):
    exp_config = None

    CONFIG_MODE = "automatic"

    if (CONFIG_MODE == "manual"):
        expNet_list = "deeplabv3_resnet101"   # fcn_resnet101 OR deeplabv3_resnet101

        ####################
        # experiment config
        ####################
        exp_config = {'lRate': 0.01,
                      'batch_size': 8,
                      'n_epochs': 200,
                      'wDecay': 0.008,
                      'classes': ["0", "1", "2"],  # , "2"
                      'early_stopping_threshold': 35,
                      #'db_path': "/caa/Projects02/vhh/public/frame_border_detection_db/",
                      #'dst_path': "/caa/Projects02/vhh/public/frame_border_detection_db/results/",
                      'db_path': "/data/share/frame_border_detection_db_v5/rgb_3class/",
                      'dst_path': "/data/share/frame_border_detection_db_v5/results/" + str(expNet_list),
                      'expTimeStamp': datetime.now().strftime("%Y%m%d_%H%M"),
                      'expType': "gray",
                      'expNet': "deeplabv3_resnet101",
                      'pre_trained_weights': "/data/share/frame_border_detection_db_v5/results/deeplabv3_resnet101/20200324_1411_gray_deeplabv3_resnet101_ExpNum_2/best_model.pth",
                      'expNum': 1,
                      'loss_metric': "cross_entropy",   #cross_entropy OR mse  OR bce_loss
                      'data_aug_flag': False
                      }
    elif(CONFIG_MODE == "automatic"):
        if (config_dict == None):
            print("ERROR: you have to select a valid config json file")
            exit()

        exp_config = config_dict

    trainThPredictor(exp_config)  # , pre_trained_path)

## https://medium.com/@arthur_ouaknine/review-of-deep-learning-algorithms-for-image-semantic-segmentation-509a600f7b57
def trainSingleExp(config_dict=None):
    exp_config = None

    CONFIG_MODE = "automatic"

    if (CONFIG_MODE == "manual"):
        expNet_list = "deeplabv3_resnet101"   # fcn_resnet101 OR deeplabv3_resnet101

        ####################
        # experiment config
        ####################
        exp_config = {'lRate': 0.01,
                      'batch_size': 8,
                      'n_epochs': 200,
                      'wDecay': 0.008,
                      'classes': ["0", "1", "2"],  # , "2"
                      'early_stopping_threshold': 35,
                      #'db_path': "/caa/Projects02/vhh/public/frame_border_detection_db/",
                      #'dst_path': "/caa/Projects02/vhh/public/frame_border_detection_db/results/",
                      'db_path': "/data/share/frame_border_detection_db_v5/rgb_3class/",
                      'dst_path': "/data/share/frame_border_detection_db_v5/results/" + str(expNet_list),
                      'expTimeStamp': datetime.now().strftime("%Y%m%d_%H%M"),
                      'expType': "gray",
                      'expNet': "deeplabv3_resnet101",
                      'pre_trained_weights': "/data/share/frame_border_detection_db_v5/results/deeplabv3_resnet101/20200324_1411_gray_deeplabv3_resnet101_ExpNum_2/best_model.pth",
                      'expNum': 1,
                      'loss_metric': "cross_entropy",   #cross_entropy OR mse  OR bce_loss
                      'data_aug_flag': False
                      }
    elif(CONFIG_MODE == "automatic"):
        if (config_dict == None):
            print("ERROR: you have to select a valid config json file")
            exit()

        exp_config = config_dict

    train(exp_config)  # , pre_trained_path)

def trainMultipleExp():
    print("train multiple experiments ... ")

    print("load training configurations ... ")
    #config_path = "/home/dhelm/VHH_Develop/pycharm_fbd/configs/experiments_higher_features/"
    #config_path = "/home/dhelm/VHH_Develop/pycharm_fbd/configs/experiments_lower_features/"
    #config_path = "/home/dhelm/VHH_Develop/pycharm_fbd/configs/experiments_lower_features_additional/"
    #config_path = "/home/dhelm/VHH_Develop/pycharm_fbd/configs/experiments_higher_features_additional/"
    config_path = "/home/dhelm/VHH_Develop/pycharm_fbd/configs/experiments_original_sota/"

    config_files_list = [os.path.join(path, name) for path, subdirs, files in os.walk(config_path) for name in files]
    config_files_list.sort()

    # select specific configuration
    #config_files_list = [config_path + "/fcn_vgg16/train_exp_1_1.json"]

    for exp_config in config_files_list:
        print("\n\n")
        print("#########################################################")
        print("start training process: " + str(exp_config))
        print("#########################################################")
        with open(exp_config) as f:
            config_dict = json.load(f)

        train(config_dict)  # , pre_trained_path)


def testSingleExp():
    expFolder = "/data/share/frame_border_detection_db_v6/results/experiments_20200404_higher_features/20200417_1200_gray_deeplabv3_mobilenet_ExpNum_1_2_cross_entropy_hf/"

    with open(expFolder + "/experiment_notes.json", 'r') as json_file:
        param_dict = json.load(json_file)

    # dst_path = param_dict['dst_path'];
    # dst_path = path
    expNet = param_dict['expNet']
    expType = param_dict['expType']
    db_path = "/data/share/frame_border_detection_db_v5/rgb_3class/"
    batch_size = param_dict['batch_size']
    classes = param_dict['classes']
    activate_lower_features = bool(param_dict['activate_lower_features'])

    print(expNet)
    print(expType)
    print(classes)

    writer = SummaryWriter(log_dir="./runs/" + "test_NEW")

    ################
    # load dataset
    ################
    trainloader, validloader = loadSegDataset(data_dir=db_path, batch_size=batch_size, expType=expType)
    print(len(trainloader) * batch_size)
    print(len(validloader) * batch_size)
    #exit()

    # Whether to train on a gpu
    train_on_gpu = torch.cuda.is_available()
    print("Train on gpu: " + str(train_on_gpu))

    # Number of gpus
    multi_gpu = False;
    if train_on_gpu:
        gpu_count = torch.cuda.device_count()
        print("gpu_count: " + str(gpu_count))
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    model = loadModel(model_arch=expNet, classes=classes, pre_trained_path=expFolder, lower_features=activate_lower_features)

    if train_on_gpu:
        model = model.to('cuda')

    if multi_gpu:
        model = nn.DataParallel(model)

    model.eval()
    with torch.no_grad():
        rocauc_score_sum = 0
        f1_sun = 0
        acc_sum = 0
        prec_sum = 0
        rec_sum = 0

        for i, sample in enumerate(validloader):
            inputs = sample['image']
            labels = sample['mask']

            ## Convert torch tensor to Variable
            inputs = Variable(inputs)
            labels = Variable(labels)

            #print(inputs.size())
            #print(labels.size())

            # If we have GPU, shift the data to GPU
            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            #outputs = model(inputs)['out']
            #outputs = F.softmax(outputs, dim=1)
            #print(torch.max(outputs))
            #print(torch.min(outputs))

            outputs = model(inputs)['out']
            outputs = F.softmax(outputs, dim=1)
            final_mask_orig = outputs.argmax(1, keepdim=True)
            final_mask = final_mask_orig.detach().cpu().float()

            labels = labels.unsqueeze(1)

            '''
            grid = torchvision.utils.make_grid(final_mask_orig)
            writer.add_image("input", grid, 0)
            grid = torchvision.utils.make_grid(labels)
            writer.add_image("mask", grid, 0)
            '''
            #exit()
            ###############################
            # calculate pixel accuracy
            ###############################

            y_pred = final_mask.numpy().ravel()
            y_true = labels.data.cpu().numpy().ravel()
            y_pred = y_pred.astype('int')
            y_true = y_true.astype('int')

            #print(len(y_pred))
            #print(len(y_true))
            prec = precision_score(y_true, y_pred, average='weighted')
            rec = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            acc = accuracy_score(y_true, y_pred)
            rocauc_score = roc_auc_score(y_true, y_pred)

            rocauc_score_sum += rocauc_score
            f1_sun += f1
            acc_sum += acc
            prec_sum += prec
            rec_sum += rec

        print('roc_auc_score: %f' % (rocauc_score_sum / len(validloader)) )
        print('F1 score: %f' % (f1_sun / len(validloader)) )
        print('acc score: %f' % (acc_sum / len(validloader)) )
        print('prec score: %f' % (prec_sum / len(validloader)) )
        print('rec score: %f' % (rec_sum / len(validloader)) )

        conf_matrix = confusion_matrix(y_true, y_pred)
        print(conf_matrix)

        #exit()

        '''
        #tmp1 = inputs[0]
        #tmp2 = outputs['out'][0] > 0.1

        
        tmp1 = tmp1.to(dtype=torch.float64)
        tmp2 = tmp2.to(dtype=torch.float64)

        res = tmp1 | tmp2
        grid = torchvision.utils.make_grid(inputs[0])
        writer.add_image("input", grid, 0)
        grid = torchvision.utils.make_grid(labels[0])
        writer.add_image("mask", grid, 0)
        grid = torchvision.utils.make_grid(outputs['out'][0] > 0.1)
        writer.add_image("output", grid, 0)
        grid = torchvision.utils.make_grid(res)
        writer.add_image("overlay", grid, 0)
        '''
        ''''''
        #grid = torchvision.utils.make_grid(outputs['out'][0] > 0.35)
        #writer.add_image("output2", grid, 0)
        #grid = torchvision.utils.make_grid(outputs['out'][0] > 0.55)
        #writer.add_image("output3", grid, 0)
        ''''''
        ''''''

    writer.close()


res5c_output = None

def res5c_hook(module, input_, output):
    global res5c_output
    res5c_output = output



def visualizeClassActivationMaps(expFolder=None):
    #expFolder = "/data/share/frame_border_detection_db_v6/results/experiments_20200404_higher_features/20200417_1417_gray_deeplabv3_squeezenet_ExpNum_1_1_cross_entropy_hf/"
    #expFolder = "/data/share/frame_border_detection_db_v6/results/experiments_20200404_higher_features/20200417_1200_gray_deeplabv3_mobilenet_ExpNum_1_2_cross_entropy_hf/"
    expFolder = "/data/share/frame_border_detection_db_v6/results/experiments_20200404_lower_features/20200417_1742_gray_deeplabv3_mobilenet_ExpNum_1_2_cross_entropy_lf/"

    if (expFolder == None):
        print("ERROR: you have to specify a valid experiment folder!")
        exit()

    TENSORBOARD_GLOBAL_LOG = True
    TENSORBOARD_IMG_LOG = True
    TENSORBOARD_SCALAR_LOGGING = True
    SAVE_FLAG = False
    CWC_FLAG = False

    experiment_notes_file = expFolder + "/experiment_notes.json"
    model_path = expFolder + "/best_model.pth"
    expName = expFolder.split('/')[-1]

    print("load experiment details ... ")
    with open(experiment_notes_file, 'r') as json_file:
        param_dict = json.load(json_file)
    print("successfully loaded ... ")

    # dst_path = param_dict['dst_path'];
    # dst_path = path

    batch_size = param_dict['batch_size']
    classes = param_dict['classes']
    db_path = param_dict['db_path']
    expNet = param_dict['expNet']
    expType = param_dict['expType']
    dim = param_dict['resized_dim']
    activate_lower_features = param_dict['activate_lower_features']
    # activate_lower_features = False

    print("\n")
    print("########################################")
    print("Summary ")
    print("########################################")
    print("batch_size: " + str(batch_size))
    print("classes: " + str(classes))
    print("db_path: " + str(db_path))
    print("expNet: " + str(expNet))
    print("expType: " + str(expType))
    print("dim: " + str(dim))
    print("activate_lower_features: " + str(activate_lower_features))
    print("########################################")
    print("\n")

    ################
    # load dataset
    ################
    trainloader, validloader, testloader = loadSegDataset(data_dir=db_path,
                                                          batch_size=batch_size,
                                                          expType=expType,
                                                          dim=dim)

    ################
    # tensorboard
    ################
    if (TENSORBOARD_GLOBAL_LOG == True):
        writer = SummaryWriter(log_dir="./runs/" + "VIS_DEBUG_test_" + str(expName))

    ################
    # setup hardware
    ################
    # Whether to train on a gpu
    train_on_gpu = torch.cuda.is_available()
    print("Train on gpu: " + str(train_on_gpu))

    # Number of gpus
    multi_gpu = False;
    if train_on_gpu:
        gpu_count = torch.cuda.device_count()
        print("gpu_count: " + str(gpu_count))
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    ################
    # load model
    ################
    model, features = loadModel(model_arch=expNet,
                                classes=classes,
                                pre_trained_path=model_path,
                                lower_features=activate_lower_features)
    print("successfully loaded ...")

    print(features)
    feat_1 = features.model[:1]
    feat_2 = features.model[:2]
    feat_3 = features.model[:3]
    feat_4 = features.model[:4]
    feat_5 = features.model[:5]
    #print(feat_3)
    #exit()

    '''
    sub = features.children()
    for child in sub:
        for a in child.children():
            print(a)
    '''

    #feat_1.layer4.register_forward_hook(res5c_hook)
    #features(some_input)


    #features(some_input)

    if train_on_gpu:
        model = model.to('cuda')

    if multi_gpu:
        model = nn.DataParallel(model)

    model.eval()
    with torch.no_grad():   #set_grad_enabled(True)
        tOverall_acc_sum = 0;
        tAvg_per_class_acc_sum = 0;
        tAvg_jacc_sum = 0;
        tAvg_dice_sum = 0;

        for i, sample in enumerate(testloader):
            inputs = sample[0]
            labels = sample[1]
            labels = labels.long()

            ## Convert torch tensor to Variable
            inputs = Variable(inputs)
            labels = Variable(labels)

            # If we have GPU, shift the data to GPU
            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # print(inputs.size())
            # print(labels.size())
            # print(inputs.dtype)
            # print(labels.dtype)

            #############################
            # generated mask
            #############################
            outputs = model(inputs)['out']
            feat1 = feat_1(inputs)
            feat2 = feat_2(inputs)
            feat3 = feat_3(inputs)
            feat4 = feat_4(inputs)
            feat5 = feat_5(inputs)

            '''
            outputs.backward()
            # pull the gradients out of the model
            gradients = model.get_activations_gradient()
            print(gradients.size())
            '''

            #exit()

            #max_feat, max_feat_idx = torch.max(feat, dim=1, keepdim=True)
            #max_feat = torch.mean(feat, dim=1, keepdim=True)
            #print(max_feat_idx.size())
            #print(max_feat.size())

            if (len(classes) == 2):
                outputs = torch.sigmoid(outputs)

            outputs = torch.softmax(outputs, dim=1)

            ########################
            # plot on tensorboard
            ########################
            if (TENSORBOARD_IMG_LOG == True and TENSORBOARD_GLOBAL_LOG == True):
                # print(tmp.shape)
                tmp = feat1[0].detach().cpu().numpy()
                tmp = torch.tensor(np.expand_dims(tmp, axis=1))
                grid = torchvision.utils.make_grid(tmp)
                writer.add_image("feat_1", grid, 0)

                tmp = feat2[0].detach().cpu().numpy()
                tmp = torch.tensor(np.expand_dims(tmp, axis=1))
                grid = torchvision.utils.make_grid(tmp)
                writer.add_image("feat_2", grid, 0)

                tmp = feat3[0].detach().cpu().numpy()
                tmp = torch.tensor(np.expand_dims(tmp, axis=1))
                grid = torchvision.utils.make_grid(tmp)
                writer.add_image("feat_3", grid, 0)

                tmp = feat4[0].detach().cpu().numpy()
                tmp = torch.tensor(np.expand_dims(tmp, axis=1))
                grid = torchvision.utils.make_grid(tmp)
                writer.add_image("feat_4", grid, 0)

                tmp = feat5[0].detach().cpu().numpy()
                tmp = torch.tensor(np.expand_dims(tmp, axis=1))
                grid = torchvision.utils.make_grid(tmp)
                writer.add_image("feat_5", grid, 0)

        if (TENSORBOARD_GLOBAL_LOG == True):
            writer.close()

def testOnFolderV2(expFolder=None, activate_gpu=True, threshold=0.5):
    #expFolder = "/data/share/frame_border_detection_db_v6/results/experiments_20200404_higher_features/20200417_1417_gray_deeplabv3_squeezenet_ExpNum_1_1_cross_entropy_hf/"
    # expFolder = "/data/share/frame_border_detection_db_v6/results/experiments_20200404_higher_features/20200417_1200_gray_deeplabv3_mobilenet_ExpNum_1_2_cross_entropy_hf/"
    expFolder = "/data/share/frame_border_detection_db_v6/results/experiments_20200404_debug/20200417_0032_gray_fcn_resnet101_ExpNum_1_1_cross_entropy_lf/"

    if(expFolder == None):
        print("ERROR: you have to specify a valid experiment folder!")
        exit()

    TENSORBOARD_GLOBAL_LOG = False
    TENSORBOARD_IMG_LOG = False
    TENSORBOARD_SCALAR_LOGGING = False
    CSV_LOGGER_FLAG = True
    SAVE_FLAG = False
    ACTIVATE_GAUSS_FLAG = True
    CWC_FLAG = False

    experiment_notes_file = expFolder + "/experiment_notes.json"
    model_path = expFolder + "/best_model.pth"
    expName = expFolder.split('/')[-1]

    print("load experiment details ... ")
    with open(experiment_notes_file, 'r') as json_file:
        param_dict = json.load(json_file)
    print("successfully loaded ... ")

    # dst_path = param_dict['dst_path'];
    # dst_path = path

    batch_size = param_dict['batch_size']
    classes = param_dict['classes']
    db_path = param_dict['db_path']
    expNet = param_dict['expNet']
    expType = param_dict['expType']
    dim = param_dict['resized_dim']
    activate_lower_features = param_dict['activate_lower_features']
    #activate_lower_features = False
    #threshold = 0.50

    print("\n")
    print("########################################")
    print("Summary ")
    print("########################################")
    print("batch_size: " + str(batch_size))
    print("classes: " + str(classes))
    print("db_path: " + str(db_path))
    print("expNet: " + str(expNet))
    print("expType: " + str(expType))
    print("dim: " + str(dim))
    print("activate_lower_features: " + str(activate_lower_features))
    print("########################################")
    print("\n")

    ################
    # load dataset
    ################
    trainloader, validloader, testloader = loadSegDataset(data_dir=db_path,
                                                          batch_size=batch_size,
                                                          expType=expType,
                                                          dim=dim)

    ################
    # tensorboard
    ################
    if (TENSORBOARD_GLOBAL_LOG == True):
        writer = SummaryWriter(log_dir="./runs/" + "FINAL_DEBUG_test_" + str(expName))

    ################
    # setup hardware
    ################
    # Whether to train on a gpu
    train_on_gpu = torch.cuda.is_available() and activate_gpu
    print("Train on gpu: " + str(train_on_gpu))

    # Number of gpus
    multi_gpu = False;
    if train_on_gpu:
        gpu_count = torch.cuda.device_count()
        print("gpu_count: " + str(gpu_count))
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    ################
    # load model
    ################
    model, features = loadModel(model_arch=expNet,
                                classes=classes,
                                pre_trained_path=model_path,
                                lower_features=activate_lower_features)
    print("successfully loaded ...")

    if train_on_gpu:
        model = model.to('cuda')
    else:
        model = model.to('cpu')

    if multi_gpu:
        model = nn.DataParallel(model)

    model.eval()
    with torch.no_grad():
        tOverall_acc_sum = 0;
        tAvg_per_class_acc_sum = 0;
        tAvg_jacc_sum = 0;
        tAvg_dice_sum = 0;

        iou_sum = 0
        dice_sum = 0

        for i, sample in enumerate(testloader):
            inputs = sample[0]
            labels = sample[1]
            labels = labels.long()

            ## Convert torch tensor to Variable
            inputs = Variable(inputs)
            labels = Variable(labels)

            # If we have GPU, shift the data to GPU
            if train_on_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                inputs = inputs.cpu()
                labels = labels.cpu()

            #print(inputs.size())
            #print(labels.size())
            #print(inputs.dtype)
            #print(labels.dtype)

            #############################
            # generated mask
            #############################
            outputs = model(inputs)['out']
            outputs = torch.sigmoid(outputs)

            outputs[outputs >= threshold] = 1
            outputs[outputs <= threshold] = 0
            outputs = torch.argmax(outputs.squeeze(), dim=1)

            mask_pred = outputs
            mask_pred = torch.squeeze(mask_pred)

            max_pred, max_pred_idx = torch.max(mask_pred, dim=1, keepdim=True)
            median_pred, median_pred_idx = torch.median(mask_pred, dim=1, keepdim=True)
            mask_pred_max = torch.squeeze(max_pred)
            mask_pred_median = torch.squeeze(median_pred)

            #print(outputs.size())
            #print(torch.unique(final_mask))
            #exit()

            #############################
            # groundtruth mask
            #############################
            mask_gt = labels

            #################################
            ## evaluate
            #################################
            if(len(classes) == 2):
                #print("2-class problem ... ")
                mask_gt[labels == 2] = 1
                bg_idx = torch.max(mask_pred)

                mask_gt[mask_gt == 0] = 255
                mask_pred[mask_pred == 0] = 255

                mask_gt[mask_gt == 1] = 0
                mask_pred[mask_pred == 1] = 0

                mask_gt[mask_gt == 2] = 0
                mask_pred[mask_pred == 2] = 0

                mask_gt[mask_gt == 255] = bg_idx
                mask_pred[mask_pred == 255] = bg_idx
            else:
                #print("3-class problem ... ")
                #print(torch.unique(labels))
                #print(torch.unique(mask_pred))
                #print(mask_pred.size())
                #continue

                mask_gt[labels == 2] = 1
                bg_idx = torch.max(mask_pred)

                mask_gt[mask_gt == 0] = 255
                mask_pred[mask_pred == 0] = 255

                mask_gt[mask_gt == 1] = 0
                mask_pred[mask_pred == 1] = 0

                mask_gt[mask_gt == 2] = 1
                mask_pred[mask_pred == 2] = 1

                mask_gt[mask_gt == 255] = bg_idx
                mask_pred[mask_pred == 255] = bg_idx

            # apply gaussian mixture model
            if(ACTIVATE_GAUSS_FLAG == True):
                a = 1



            #exit()
            tOverall_acc_sum, tAvg_per_class_acc_sum, tAvg_jacc_sum, tAvg_dice_sum = calculateMetrics(mask_gt, mask_pred,
                                                                                                      len(classes),
                                                                                                      tOverall_acc_sum,
                                                                                                      tAvg_per_class_acc_sum,
                                                                                                      tAvg_jacc_sum,
                                                                                                      tAvg_dice_sum)
            #print(len(testloader))
            #print(tAvg_jacc_sum / len(testloader))
            #print(tAvg_dice_sum / len(testloader))

            #exit()
            ###############################
            # write results to tensorboard
            ###############################
            if (TENSORBOARD_SCALAR_LOGGING == True and TENSORBOARD_GLOBAL_LOG == True):
                writer.add_scalar('test_overall_acc', tOverall_acc_sum / len(testloader), i)
                writer.add_scalar('test_per_class_acc', tAvg_per_class_acc_sum / len(testloader), i)
                writer.add_scalar('test_Avg_jacc', tAvg_jacc_sum / len(testloader), i)
                writer.add_scalar('test_Avg_dice', tAvg_dice_sum / len(testloader), i)

            ########################
            # plot on tensorboard
            ########################
            if(SAVE_FLAG == True):

                img_np = inputs.detach().cpu().numpy()
                mask_pred_np = np.expand_dims(mask_pred.detach().cpu().numpy(), axis=1)
                mask_gt_np = np.expand_dims(labels.detach().cpu().numpy(), axis=1)

                for a in range(0, len(mask_gt)):
                    img = np.reshape(img_np[a], (dim[0], dim[1], 3))
                    m_p = np.reshape(mask_pred_np[a], (dim[0], dim[1], 1))
                    m_gt = np.reshape(mask_gt_np[a], (dim[0], dim[1], 1))

                    m_p[m_p == 1] = 255
                    m_p[m_p == 2] = 255

                    m_gt[m_gt == 1] = 255
                    m_gt[m_gt == 2] = 255

                    cv2.imwrite("./templates/imgs/img_" + str(a + 1) + "_" + str(i + 1) + ".png", img)
                    cv2.imwrite("./templates/mask_preds/s_" + str(a + 1) + "_" + str(i + 1) + ".png", m_p)
                    cv2.imwrite("./templates/mask_gts/gt_" + str(a + 1) + "_" + str(i + 1) + ".png", m_gt)

            ########################
            # plot on tensorboard
            ########################
            if (TENSORBOARD_IMG_LOG == True and TENSORBOARD_GLOBAL_LOG == True):
                #print(tmp.shape)
                grid = torchvision.utils.make_grid(inputs.detach().cpu())
                writer.add_image("input", grid, 0)

                tmp = np.expand_dims(labels.detach().cpu().numpy(), axis=1)
                grid = torchvision.utils.make_grid(torch.tensor(tmp))
                writer.add_image("mask_gt", grid, 0)

                tmp = np.expand_dims(mask_pred.detach().cpu().numpy(), axis=1)
                grid = torchvision.utils.make_grid(torch.tensor(tmp))
                writer.add_image("mask_pred", grid, 0)

                #tmp = np.expand_dims(mask_pred_median.detach().cpu().numpy(), axis=1)
                #grid = torchvision.utils.make_grid(torch.tensor(tmp))
                #writer.add_image("mask_pred_median", grid, 0)

                #tmp = np.expand_dims(mask_pred_max.detach().cpu().numpy(), axis=1)
                #grid = torchvision.utils.make_grid(torch.tensor(tmp))
                #writer.add_image("mask_pred_max", grid, 0)


    print("tOverall_acc: " + str(tOverall_acc_sum / len(testloader)))
    print("tAvg_per_class_acc: " + str(tAvg_per_class_acc_sum / len(testloader)))
    print("tAvg_jacc: " + str(tAvg_jacc_sum / len(testloader)))
    print("tAvg_dice: " + str(tAvg_dice_sum / len(testloader)))

    ###############################
    # write results to csv
    ###############################
    if (CSV_LOGGER_FLAG == True):
        entries_list = []
        entries_list.append(float(tOverall_acc_sum / len(testloader)))
        entries_list.append(float(tAvg_per_class_acc_sum / len(testloader)))
        entries_list.append(float(tAvg_jacc_sum / len(testloader)))
        entries_list.append(float(tAvg_dice_sum / len(testloader)))

        path = pathlib.Path(expFolder).parent
        csvLogger(dst_folder=str(path),
                  name="final_tests_metrics_" + str(path).split('/')[-1] + "_th-" + str(threshold) + ".csv",
                  epoch=0,
                  expName=expName,
                  entries_list=entries_list);

    if (TENSORBOARD_GLOBAL_LOG == True):
        writer.close()

def runMultipleTests():
    #root_dir = "/data/share/frame_border_detection_db_v6/results/experiments_20200404_debug/"
    #root_dir = "/data/share/frame_border_detection_db_v6/results/experiments_20200404_higher_features"
    #root_dir = "/data/share/frame_border_detection_db_v6/results/experiments_20200404_lower_features"
    root_dir = "/data/share/frame_border_detection_db_v6/results/experiments_original_sota/"

    threshold = 0.50

    folder_name = root_dir.split('/')[-1]
    #exp_dir_list = os.listdir(root_dir)
    #print(exp_dir_list)
    #print(folder_name)

    ################
    # delete old file
    ################
    if (os.path.exists(root_dir + "/final_tests_metrics_" + "_th-" + str(threshold) + ".csv")):
        print("delete old file ... ")
        os.remove(root_dir + "/final_tests_metrics_" + str(folder_name) + "_th-" + str(threshold) + ".csv")
    exp_dir_list = os.listdir(root_dir)
    exp_dir_list = [name for name in exp_dir_list if os.path.isdir(os.path.join(root_dir, name))]

    ################
    # setup results log file
    ################
    CSV_LOGGER_FLAG = True
    if (CSV_LOGGER_FLAG == True):
        entries_list = []
        entries_list.append(str("tOverall_acc"))
        entries_list.append(str("tAvg_per_class_acc"))
        entries_list.append(str("tAvg_jacc"))
        entries_list.append(str("tAvg_dice"))

        csvLogger(dst_folder=root_dir,
                  name="final_tests_metrics_" + str(folder_name) + ".csv",
                  epoch=0,
                  expName="expName",
                  entries_list=entries_list);

    for exp_name in exp_dir_list:
        full_path = os.path.join(root_dir, exp_name)

        if not os.path.exists(full_path):
            print("WARNING: not results for this experiment!")
            continue

        print("\n\n")
        print("#########################################################")
        print("start test process: " + str(full_path))
        print("#########################################################")
        testOnFolderV2(expFolder=full_path, threshold=threshold)





def calculateDataBaseStatistics():
    db_path = "/data/share/frame_border_detection_db_v6/rgb_3class_large/"
    dst_path = "/data/share/frame_border_detection_db_v6/"

    trainloader, validloader, testloader = loadSegDataset(data_dir=db_path, batch_size=1, dim=(180, 240))

    all_samples_r_np = []
    all_samples_g_np = []
    all_samples_b_np = []
    for i, sample in enumerate(trainloader):
        inputs = sample[0]
        labels = sample[1]

        img = inputs.detach().cpu().numpy()[0]
        img = np.reshape(img, (img.shape[2], img.shape[1], img.shape[0]))
        #print(img.shape)

        dim = (180, 240);
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA);
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        r, g, b = cv2.split(img);

        all_samples_r_np.append(r)
        all_samples_g_np.append(g)
        all_samples_b_np.append(b)

    all_samples_r_np = np.array(all_samples_r_np)
    all_samples_g_np = np.array(all_samples_g_np)
    all_samples_b_np = np.array(all_samples_b_np)

    mean_r, mean_g, mean_b = calculateMean(all_samples_r_np, all_samples_g_np, all_samples_b_np);
    std_r, std_g, std_b = calculateSTD(all_samples_r_np, all_samples_g_np, all_samples_b_np);

    # save statiscits
    saveStatistics(dst_path, 240, mean_r, mean_g, mean_b, std_r, std_g, std_b);

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True,
                help="-m, --mode --> select execution mode (e.g. train, test")
    ap.add_argument("-c", "--config", required=False,
                    help="-c, --config --> specify valid json configuration")
    args = vars(ap.parse_args())
    mode = str(args["mode"]);
    config_path = str(args["config"]);

    if (mode == "train_single"):
        print("INFO: train_single experiment is selected ... ")

        with open(config_path) as f:
            config_dict = json.load(f)

        # Create the dataloader
        trainSingleExp(config_dict=config_dict);

    elif (mode == "train_own"):
        print("INFO: train_own experiment is selected ... ")

        with open(config_path) as f:
            config_dict = json.load(f)

        # Create the dataloader
        trainSingleExpThPredictor(config_dict=config_dict);
    elif (mode == "train_multiple"):
        print("INFO: train_multiple experiment is selected ... ")
        trainMultipleExp()
    elif (mode == "test_single"):
        print("INFO: test_single experiment is selected ... ")
        testSingleExp()
    elif (mode == "test_multiple"):
        print("INFO: test_multiple experiment is selected ... ")
        runMultipleTests()
    elif (mode == "calculate_statistics"):
        print("INFO:calculate_statistics is selected ... ")
        calculateDataBaseStatistics()
    elif (mode == "test_folder"):
        print("INFO:test on folder selected ... ")
        testOnFolderV2()
    elif (mode == "debug"):
        print("INFO: DEBUG MODE ... ")
        visualizeClassActivationMaps()
    else:
        print("ERROR: select valid option!")
        print("usage: blablablabla\n"
              "Examples:\n"
              "python others/testPyTorchClassifier.py -m train_own\n"
              "python others/testPyTorchClassifier.py -m train_single\n"
              "python others/testPyTorchClassifier.py -m train_multiple\n"
              "python others/testPyTorchClassifier.py -m test_single\n"
              "python others/testPyTorchClassifier.py -m test_multiple\n")

main()
