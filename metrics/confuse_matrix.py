
import warnings
from typing import Union

import numpy as np
from sklearn import metrics
import torch

from monai.utils import Average, look_up_option

from monai.metrics import CumulativeIterationMetric


class MultiConfuseMatrixMetric(CumulativeIterationMetric):
    
    def __init__(self, argmax=True, threshold=0.5) -> None:
        super().__init__()
        self.argmax = argmax
        self.threshold = threshold
        
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        return y_pred, y
    
    def aggregate(self, average: Union[Average, str, None] = None):
        """
        Typically `y_pred` and `y` are stored in the cumulative buffers at each iteration,
        This function reads the buffers and computes the area under the ROC.

        Args:
            average: {``"macro"``, ``"weighted"``, ``"micro"``, ``"none"``}
                Type of averaging performed if not binary classification. Defaults to `self.average`.

        """
        y_pred, y = self.get_buffer()
        # compute final value and do metric reduction
        if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError("y_pred and y must be PyTorch Tensor.")
        if self.argmax:
            y_pred = [i.cpu().argmax(0) for i in y_pred]
        else:
            y_pred = [torch.softmax(i.cpu(), dim=0)[0] > self.threshold for i in y_pred]
        y = [i.cpu().argmax(0) for i in y]
        
        cm = metrics.confusion_matrix(np.stack(y, axis=0), np.stack(y_pred, axis=0))
        # roc_auc_dict = roc_auc_score_multiclass(np.stack(y_pred, axis=0), np.stack(y, axis=0), average)
        return torch.tensor(cm)
    

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

    #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = metrics.roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict
    