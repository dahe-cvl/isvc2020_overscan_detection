import numpy as np
import os
from utils import *

root_path = "/data/share/frame_border_detection_db_v6/results/experiments_20200404_higher_features"
experiment_list = os.listdir(root_path)

experiment_list = [name for name in experiment_list if os.path.isdir(os.path.join(root_path, name))]

loss_history_all = np.zeros((51, 2))
for exp in experiment_list:
    experiment_path = os.path.join(root_path, exp)
    metrics_log_path = os.path.join(experiment_path, "metrics_history.log")
    entries_np = csvReader(metrics_log_path)
    loss_history = np.round(entries_np[:, 10:12].astype('float32'), 5)
    header = np.array([[exp + "_val_jacc", exp + "_val_dice"]])
    print(header.shape)
    print(loss_history.shape)

    if(len(loss_history) != len(loss_history_all)):
        tmp = np.zeros((loss_history_all.shape[0] - 1, loss_history.shape[1]))
        #print(tmp.shape)
        #print(tmp)
        tmp[:loss_history.shape[0]] = loss_history
        loss_history = tmp
        #print(loss_history.shape)
    #print(tmp.shape)
    #print(tmp)
    #print(loss_history.shape)
    #exit()

    loss_history = np.concatenate((header, loss_history), axis=0)
    print(loss_history.shape)

    loss_history_all = np.concatenate((loss_history_all, loss_history), axis=1)

    print(loss_history.shape)
    print(loss_history_all.shape)

loss_history_all = loss_history_all[:, 2:]

print("----------------------------------")
#print(experiment_path)
#print(loss_history)
print(loss_history_all)

for i in range(0, len(loss_history_all)):
    entry = loss_history_all[i].tolist()
    csvWriter(dst_folder="../", name="summary_hf_dice.csv", entries_list=entry)

#exit()