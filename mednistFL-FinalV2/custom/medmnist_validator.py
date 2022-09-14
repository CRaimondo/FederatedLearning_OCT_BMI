# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import os
import PIL
from PIL import Image

import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from medmnist_dataset_class import MedMNIST
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision import models as models
import torchvision.transforms as transforms
from datetime import date
import warnings
warnings.simplefilter("ignore", UserWarning)
from sklearn import metrics
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#
from nvflare.apis.dxo import from_shareable, DataKind, DXO
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from torchvision import models as models
import matplotlib.pyplot as plt
from itertools import cycle

class medmnist_Validator(Executor):
    
    def __init__(self, validate_task_name=AppConstants.TASK_VALIDATION):
        super(medmnist_Validator, self).__init__()

        self._validate_task_name = validate_task_name

        batch_size = 64
        in_channels = 1
        num_classes = 4

        # Define the model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=False, progress = True)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride= (2,2) , padding=(3, 3), bias=False)
        #self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride= (1,1) , padding=(1, 1), bias=False)
        self.model.fc = nn.Linear(512, num_classes)
        self.model.to(self.device)
        
        # Point to the relevent test label data annd jpeg images
        val_csv_path = './input/vallabels_site1_SUBSET25.csv'
        image_path = './input/octmnist_images'

        data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])


        self._val_dataset = MedMNIST(val_csv_path, image_path, transform = data_transform, as_rgb = True)
        self._val_loader = DataLoader(self._val_dataset, batch_size=batch_size, shuffle=False)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

                ### Load validation measures from do_validation function, can add more metrics here
                validation_results = self.do_validation(weights, abort_signal)
                auc_score = validation_results['AUC']
                acc_ = validation_results['ACC']



                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(fl_ctx, f"\n \n++++++++++++++++++++++++\n"
                                      f"AUC and ACC when validating {model_owner}'s model on"
                                      f" {fl_ctx.get_identity_name()}"f's data: {auc_score},{acc_}')
                print('++++++++++++++++++\n')

                dxo = DXO(data_kind=DataKind.METRICS, data={'val_accuracy': acc_,
                                                             'val_AUC':auc_score}) # validation_results go here
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)


    ## Generates Run Validation Loop and Generated Metrics

    def do_validation(self, weights, abort_signal):
        self.model.load_state_dict(weights)     # Key line, brings in model weights from cross-sites
        self.model.eval()

        running_correct = 0
        running_total = 0
        y_pred = []
        y_true = []
        y_scores = torch.tensor([])
        y_targets = torch.tensor([])
        y_test = []
        
        classes_list = ['cnv','dme','drusen','healthy']

        with torch.no_grad():
            for i, batch in enumerate(self._val_loader):
                if abort_signal.triggered:
                    return 0

                image = batch['image'].to(self.device)
                label = batch['label'].to(self.device)
                target = batch['label'].to(self.device)
                output = self.model(image)
                y_binarized = label_binarize(target.cpu(), classes = [0,1,2,3]) # sklearn function to binarize labels for multi-class
                y_score = output.softmax(dim=-1).cpu()
                y_scores = torch.cat((y_scores, y_score),0)

                if y_test == []:
                    y_test = y_binarized
                else:
                    y_test = np.concatenate((y_test, y_binarized),axis = 0)

              
                _ , preds = torch.max(output,dim=1)

                y_true += label.cpu().numpy().tolist() # true label list
                y_pred += preds.cpu().tolist() # predicted label lists

                targets = target.squeeze().long().cpu()
                targets = targets.float().resize_(len(targets), 1)
                y_targets = torch.cat((y_targets, targets), 0)

                num_img = image.size()[0]
                running_correct += torch.sum(preds == label).item()
                running_total += num_img
                
            y_targets = y_targets.detach().numpy()
            y_scores = y_scores.detach().numpy()


        metrics_output = {}

        metrics_output = self.get_val_metrics(y_true, y_scores)

        return metrics_output
        
        # if want to return a metric ->  metrics_output['AUC'] 

        #### end of do_validation


    def get_val_metrics(self, y_true, y_scores):
        val_AUC = self.getAUC(y_true, y_scores)
        val_ACC = self.getACC(y_true, y_scores)
        
        return {'AUC':val_AUC, 'ACC':val_ACC}

    def getAUC(self, y_true, y_scores):
        '''AUC metric.
        :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
        :param y_score: the predicted score of each class,
        shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
        :param task: the task of current dataset
        '''
        y_true = np.array(y_true) # from list to np array, good
        y_true = y_true.squeeze()
        y_scores = y_scores.squeeze()

        auc = 0
        for i in range(y_scores.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_scores[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        AUC = auc / y_scores.shape[1]
        return AUC


    def getACC(self, y_true, y_scores):
        '''Accuracy metric.
        :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
        :param y_score: the predicted score of each class,
        shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
        :param task: the task of current dataset
        :param threshold: the threshold for multilabel and binary-class tasks
        '''
        y_true = np.array(y_true)
        y_true = y_true.squeeze()
        y_scores = y_scores.squeeze()

        ACC = accuracy_score(y_true, np.argmax(y_scores, axis=-1))

        return ACC


                
        # for i in range(len(classes_list)):
        #     print(f'\nMetrics for subtype: {label_list[i]}')
        #     subtype_labels=np.array(running_labels[:, i]).flatten()
        #     subtype_outputs=np.array(running_outputs[:, i]).flatten()

        #     # Calculate ROC AUC for subtype
        #     fpr, tpr, _ = metrics.roc_curve(subtype_labels, subtype_outputs)
        #     roc_auc = round(metrics.auc(fpr, tpr), 5)
        #     print(f'ROC_auc: {roc_auc}')
        #     # Caclulate PRC AUC for subtype
        #     precision, recall, thresholds = metrics.precision_recall_curve(subtype_labels, subtype_outputs)
        #     prc_auc = round(metrics.auc(recall, precision), 5)
        #     print(f'PRC_auc: {prc_auc}')
        #     # Binarize predictions
        #     bin_threshold = 0.4
        #     bin_output = np.where(subtype_outputs > bin_threshold, 1, 0)
        #     # Calculate accuracy score using threshold for binary
        #     acc = round(metrics.accuracy_score(subtype_labels, bin_output), 5)
        #     # Calculate F1 score using threshold for binary
        #     f1 = round(metrics.f1_score(subtype_labels, bin_output), 5)
        #     print(f'F1 score with bin_thresh of {bin_threshold}: {f1}')

        #     metrics_output[label_list[i]]=(roc_auc, prc_auc, f1)

        # return metrics_output
        

       