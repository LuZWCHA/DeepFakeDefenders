# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
 
# name_list = ['Monday','Tuesday','Friday','Sunday']
# num_list = [1.5,0.6,7.8,6]
# num_list1 = [1,2,3,1]
# x =list(range(len(num_list)))
# total_width, n = 0.8, 2
# width = total_width / n
 
# plt.bar(x, num_list, width=width, label='boy',fc = 'y')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, num_list1, width=width, label='girl',tick_label = name_list,fc = 'r')
# plt.legend()
# plt.show()

import numpy as np
import pandas as pd
# df = pd.read_csv("/nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/work_dir/classify/results.csv")
# df = pd.read_csv("/nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/work_dir/EfficientNet_B2_NPR_GRAD_deepfake_c2_testset/results.csv")
df = pd.read_csv("/nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/_merged_av_res.csv")

if "pred" in df:
    df["pred_logits"] = df["pred"]
if "id" in df:
    df["file_info"] = df["id"]

df = df[["file_info", "gt", "pred_logits"]]
print(df.head())
# df["real"] = df["pred_logits"].values[:, 0]
# print(df["pred_logits"].values)
try:
    df['pred_logits'] = df['pred_logits'].apply(eval).apply(np.array)
except:
    pass
# print(df["pred_logits"].values.to_numpy())
df['fake'] = df['pred_logits'].apply(lambda x: x[1] if not isinstance(x, float) else x)
df["fake_true"] = df['fake'][df['gt'] == 1]
df["fake_false"] = df['fake'][df['gt'] == 0]

plt.hist(df[["fake_true", "fake_false"]], bins=40, range=(0, 1))
plt.show()
print(df)

df.to_csv("res.csv")