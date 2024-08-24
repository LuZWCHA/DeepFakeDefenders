from matplotlib import axes
import pandas as pd

_pd1 = pd.read_csv("/nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/val_submit.csv")
_pd2 = pd.read_csv("/nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/val_submit_vit_b_cls_token_mixer_epoch=3-step=20166.csv")

_pd1 = pd.merge(_pd1, _pd2, on="video_name")
print(_pd1)
_pd1["y_pred"] = _pd1.apply(lambda x: min(x["y_pred_x"], x["y_pred_y"]), axis=1)
print(_pd1)
_pd1 = _pd1[["video_name", "y_pred"]]
_pd1.to_csv("_merged_min_op.csv", index=False)

