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

import enum
import os.path
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision import models as models
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from medmnist_dataset_class import MedMNIST
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from datetime import date
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from skimage import io
from skimage import color
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import cycle
import PIL
import warnings
warnings.simplefilter("ignore", UserWarning)
from sklearn.metrics import  confusion_matrix, roc_curve, auc
import itertools  


from nvflare.apis.dxo import from_shareable, DXO, DataKind, MetaKey
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager
from pt_constants import PTConstants


class medmnist_Trainer(Executor):

    def __init__(self, lr=0.01, epochs=6, train_task_name=AppConstants.TASK_TRAIN,
                 submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL, exclude_vars=None, data_flag = "octmnist"):
        """
        Args:
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            train_task_name (str, optional): Task name for train task. Defaults to "train".
            submit_model_task_name (str, optional): Task name for submit model. Defaults to "submit_model".
            exclude_vars (list): List of variables to exclude during model loading.
        """
        super(medmnist_Trainer, self).__init__()
        #
        self._lr = lr
        self._epochs = epochs
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars
        self._data_flag = data_flag

        # Training setup

        # Define MedNIST Dataset
        if self._data_flag == 'octmnist':
            in_channels = 1
            num_classes = 4
        elif self._data_flag == 'retinamnist':
            in_channels = 3
            num_classes = 5
        else:
            in_channels =3
            num_classes = 6

        # Define the model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=False, progress = True)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride= (2,2) , padding=(3, 3), bias=False)
        #self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride= (1,1) , padding=(1, 1), bias=False)
        self.model.fc = nn.Linear(512, num_classes)
        self.model.to(self.device)

        # Define Training Parameters: batch size, loss fn, and optimizer, scheduler
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        batch_size = 64

        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        # Point to image and label data
        train_csv_path = './input/trainlabels_site1_SUBSET25.csv'  # make sure each site has own csv
        val_csv_path = './input/vallabels_site1_SUBSET25.csv'
        image_path = './input/octmnist_images'


        # Load datasets using dataset class with desired pixel size and transforms 

        # define transforms - not sure if needed here or can keep only in dataset_class.py
        data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                 ])

        self._train_dataset = MedMNIST(train_csv_path, image_path, transform = data_transform, as_rgb = True)
        self._val_dataset = MedMNIST(val_csv_path, image_path, transform = data_transform, as_rgb = True)

        self._train_loader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True, 
                                                    num_workers = 0, drop_last = False, pin_memory = False)

        self._val_loader = DataLoader(self._val_dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers = 0, drop_last = False, pin_memory = False)
        
        self._dataloader = {'train': self._train_loader, 'local_validate': self._val_loader}

        self._n_iterations = len(self._train_loader)

        # define weighted loss for imabalanced dataset
        #loss_weights = self._train_dataset.loss_weights.to(self.device)
        #print(f'\nloss weights: {loss_weights}\n')
        
        self.loss_fn = nn.CrossEntropyLoss()

        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager
        # in case no initial model is found.
        self._default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf)

    def local_train(self, fl_ctx, weights, abort_signal):
        print('\nStarting Training...\n')
        # Set the model weights
        self.model.load_state_dict(state_dict=weights)

        acc_dict = {'train':[], 'local_validate':[]}
        loss_dict = {'train':[], 'local_validate':[]}
        best_acc = 0
        phases = ['train','local_validate']
        y_pred = []
        y_true = []

        y_scores = torch.tensor([])
        y_test = [] # binarized labels
    
    
        local_output_dir = self.create_output_dir(fl_ctx)   

        for i in tqdm(range(self._epochs)):
            print(f'Epoch {i+1} of {self._epochs}\n')
            print('-'*20)

            for p in phases:
                running_correct = 0
                running_loss = 0
                running_total = 0

                if p == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                
                for batch in self._dataloader[p]:
                    if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                        return
                    
                    self.optimizer.zero_grad()
                    image = batch['image'].to(self.device)
                    label = batch['label'].to(self.device)
                    #target = batch['label'].to(self.device)

                
                    output = self.model(image)
                    loss = self.loss_fn(output,label)
                    _ , preds = torch.max(output,dim=1)
                    num_img = image.size()[0]
                    running_correct += torch.sum(preds == label).item()
                    running_loss += loss.item()* num_img
                    running_total += num_img

                    if p == 'train':
                        loss.backward()
                        self.optimizer.step()

                    if p ==  'local_validate':
                        y_binarized = label_binarize(label, classes = [0,1,2,3]) #label instead of target here to clean up
                        y_score = output.softmax(dim=-1)
                        y_scores = torch.cat((y_scores, y_score),0)
                        y_true += label.tolist()

                        if y_test == []:
                            y_test = y_binarized
                        else:
                            y_test = np.concatenate((y_test, y_binarized), axis = 0)

                        #if i == (self._epochs-1):
                        y_pred += preds.tolist() # dont use y_pred
                            #y_true += label.tolist()

                        
                epoch_acc = float(running_correct/running_total)
                epoch_loss = float(running_loss/running_total)

                acc_dict[p].append(epoch_acc)
                loss_dict[p].append(epoch_loss)

                #if (i%10 == 0): only doing the start of training
                print('\nPhase:{}, Epoch Loss: {:.4f} | Epoch Acc: {:.4f}\n'.format(p, np.average(loss_dict[p]), epoch_acc))

        

                if p == 'local_validate':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        #best_model_wts = self.model.state_dict()
                    #y_test = y_test.numpy()
                    #y_scores = y_scores.detach().numpy()

        print('Best val acc: {:4f}\n'.format(best_acc))

        # End of Trainning Loop


        # Get Trainning and Local Validation Metrics 

        local_AUC = self.getAUC(y_true,y_scores)

        # Plot metrics
        self.plot_metrics(local_output_dir, loss_dict['train'], loss_dict['local_validate'], 'Epoch Loss')
        self.plot_metrics(local_output_dir, acc_dict['train'], acc_dict['local_validate'], 'Epoch Accuracy')

        # Plot local val ROC
        self.plot_roc_curve(local_output_dir, y_test,y_scores, num_classes= 4, title_string = 'One-Vs-Rest Multi-Class ROC')

        # Plot Confusion Matrix
        self.cm(local_output_dir, y_true,y_pred, title_string = 'Confusion Matrix')

        class_names = ['0:cnv','1 : dme','2 : drusen','3: healthy']

        # Get Classification Report

        report = self.get_classificationreport(y_true, y_pred, class_names)
        report.to_csv(f'{local_output_dir}/classificationreport_metrics.csv', index = True)

        ## Save metrics to csv 
        print(f'saving to {local_output_dir}/epoch_metrics.csv')
        epoch_metrics = pd.DataFrame(data = {'epoch':list(range(i + 1)),
                                        'train_loss': loss_dict['train'],
                                        'train_acc':acc_dict['train'],
                                        'val_loss':loss_dict['local_validate'], 
                                        'val_acc':acc_dict['local_validate'],
                                        'val AUC':local_AUC})
        
        epoch_metrics.to_csv(f'{local_output_dir}/epoch_metrics.csv', index = False)

     
    def plot_metrics(self, output_dir, train_data, val_data, title_string):
        # Plot
        fig, axs = plt.subplots(figsize = (16,8))
        axs.plot(train_data, color = 'blue', label='Train Results')
        axs.plot(val_data, color = 'orange', label='Validation Results')
        axs.set_title(f"{title_string}")
        axs.legend(loc='best')
        print(f"\nplotting {title_string}...\n")
        fig.savefig(f'{output_dir}/{title_string}.png')
        return


    def plot_roc_curve(self,output_dir, y_test, y_scores, num_classes, title_string):
        plt.clf()
        y_test = y_test.squeeze()

        y_scores =  y_scores.detach().numpy()
        y_scores = y_scores.squeeze()
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = num_classes   # change per task

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'mediumorchid', 'lightcoral'])
        lw = 2

        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                        label='ROC curve of class {0} (area = {1:0.2f})'
                        ''.format(i, roc_auc[i]))
           # print('AUC of Class',i, roc_auc[i])
        
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('One-Vs-Rest Multi-Class ROC')
        plt.legend(loc="lower right")
        plt.savefig(f'{output_dir}/{title_string}.png')
        return
            
    def getAUC(self, y_true, y_scores):
        '''AUC metric.
        :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
        :param y_score: the predicted score of each class,
        shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
        :param task: the task of current dataset
        '''
        y_true = np.array(y_true) # from list to np array, good
        y_true = y_true.squeeze()
        y_scores =  y_scores.detach().numpy() 
        y_scores = y_scores.squeeze()

        auc = 0
        for i in range(y_scores.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_scores[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        AUC = auc / y_scores.shape[1]
        return AUC
    
    def plot_confusion_matrix(output_dir, title_string, cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f'{output_dir}/{title_string}.png')

    # Compute confusion matriix
    def cm(self, output_dir, y_true, y_pred, title_string):
        y_true = np.array(y_true) # from list to np array, good
        #y_true = y_true.squeeze()
        y_pred =  np.array(y_pred)
        #y_scores = y_scores.squeeze()

        plt.clf()
        cnf_matrix = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        plt.figure()
        class_names = ['0:cnv','1 : dme','2 : drusen','3: healthy']  # change per task
        classes =class_names 
        cmap=plt.cm.Blues
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f'
        thresh = cnf_matrix.max() / 2.
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, format(cnf_matrix[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cnf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f'{output_dir}/{title_string}.png')




    def get_classificationreport(self, y_true, y_pred, class_names):
        y_true = np.array(y_true) # from list to np array, good
        #y_true = y_true.squeeze()
        y_pred =  np.array(y_pred)
        
        report = pd.DataFrame(classification_report(y_true, y_pred, target_names = class_names, output_dict=True)).transpose()
        return report
      
    


    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            if task_name == self._train_task_name:
                # Get model weights
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Convert weights to tensor. Run training
                torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
                self.local_train(fl_ctx, torch_weights, abort_signal)

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Save the local model after training.
                self.save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                new_weights = self.model.state_dict()
                new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

                outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=new_weights,
                                   meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations})
                return outgoing_dxo.to_shareable()
            elif task_name == self._submit_model_task_name:
                # Load local model
                ml = self.load_local_model(fl_ctx)

                # Get the model parameters and create dxo from it
                dxo = model_learnable_to_dxo(ml)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except:
            self.log_exception(fl_ctx, f"Exception in simple trainer.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def create_output_dir(self, fl_ctx: FLContext):
        # Make directory named "output" in current run_# folder
        # Get run number dir path
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        output_dir = os.path.join(run_dir, PTConstants.OutputMetricsDir)
        # make output folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        round_int = 1
        round_output_dir = os.path.join(run_dir, PTConstants.OutputMetricsDir, f"round_{round_int}")
        while os.path.exists(round_output_dir):
            round_int += 1
            round_output_dir = os.path.join(run_dir, PTConstants.OutputMetricsDir, f"round_{round_int}")
        os.makedirs(round_output_dir)
        print('\ncreating output dir...')
        print(round_output_dir)
        return round_output_dir



    def save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

    def load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(data=torch.load(model_path),
                                                                   default_train_conf=self._default_train_conf)
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        return ml