import numpy as np
import torch, pandas as pd

class Results:
    def __init__(self):
        pass
        self.dataset = {
            "file_info": [],
            "pred_onehot": [],
            "pred_logits": [],
            "gt": [],
        }
    
    def _tolist(self, tensor_ndarray):
        # print(tensor_ndarray)
        if isinstance(tensor_ndarray, str) or isinstance(tensor_ndarray, (int, float)):
            return tensor_ndarray
        if isinstance(tensor_ndarray, torch.Tensor):
            tensor_ndarray = tensor_ndarray.cpu().numpy()
        if isinstance(tensor_ndarray, list):
            tensor_ndarray = [self._tolist(i) for i in tensor_ndarray]
        
        # if isinstance(tensor_ndarray, (list, str)):
        #     return tensor_ndarray
        if isinstance(tensor_ndarray, np.ndarray):
            shape = tensor_ndarray.shape
            assert len(shape) in [0, 1, 2], f"Not support size > 2, got shape {shape}"
            if len(shape) == 0:
                return tensor_ndarray.item()
            if len(shape) <= 1:
                tensor_ndarray = tensor_ndarray[:, None]
            
        res = []
        for i in tensor_ndarray:
            if isinstance(i, np.ndarray):
                i = i.tolist()
            if isinstance(i, list) and len(i) == 1:
                i = i[0]
            res.append(i)
        # print(res)
        return res
    
    def process(self, pred_onehot, pred_logits, gt, file_info):
        # print(gt.shape)
        pred_onehot = self._tolist(pred_onehot)
        pred_logits = self._tolist(pred_logits)
        gt = self._tolist(gt)
        file_info = self._tolist(file_info)
        
        self.dataset["file_info"].extend(file_info)
        self.dataset["pred_onehot"].extend(pred_onehot)
        self.dataset["pred_logits"].extend(pred_logits)
        self.dataset["gt"].extend(gt)
    
    def reset(self):
        self.dataset = {
            "file_info": [],
            "pred_onehot": [],
            "pred_logits": [],
            "gt": [],
        }
    
    def save(self, path):
        for k, v in self.dataset.items():
            print(k, len(v))
        pd.DataFrame(self.dataset).to_csv(path, index=False)