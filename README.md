# Deepfake Video
This is naf team's codes.

## Software Environment

### System information 
docker image: [11.6.2-cudnn8-devel-ubuntu20.04](https://hub.docker.com/r/nvidia/cuda/tags?page=&page_size=&ordering=&name=11.6.2-cudnn8-devel-ubuntu20.04)
``` bash
docker pull nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
```
### Conda environment
> Note: 最好手动安装pytorch-gpu保证可复现性。
``` bash
# Please install torch manually
conda env create -f environment.yml
# Swich to deepfake env
conda activate deepfake
```

## GPUs
A100 80G x 2

## Train
This part will describ the way to reproduce the model training process.
My model trained effect from the server souce lack, I trained the model follow to speed up my experiments:
1. Pretrain a model 1 epoch on trainset with any aguments [epoch=0 step=1260];
2. Load the model as pretrained, and add many different transforms (Audio-Visual offset and so on), train 5 epoch


### Dataset Structure

``` bash
path/to/dataset
|-- train.csv
|-- trainset
|   |-- 02cbac6f98b8a9a33bf8206b88b11417.mp4
|   |-- 52cbac6f98b8a92d3bf8206b88b11417.mp4
|-- val.csv
|-- valset
|   |-- 02cd793e3ecac106c377cb52a6d2f586.mp4
|   |-- 02d03ae939d0302cce05cdc9bf1c7cd2.mp4
...
```

The **csv** file contents looks like:
```sql
video_name,target
4663658dbe76be9de4d616b573546a0c.mp4,0
8fedbd1856eb89741d7488247f05d7e8.mp4,0
5a5275506a58ed0b27e3288c042a6d9e.mp4,0
7d27e58b3a8273a7ee5f9efd5440fd4f.mp4,0
...
```

### Run Scrips

***Recommend!***
``` bash
python train_video_demo.py path/to/dataset_root -m train
```
\
Or run the original train file (in my exp stage) ```train_video.py```
(Please **modify** the **dataset path** [see #DeepFakeDataModule._DATA_PATH])
``` bash
python train_video.py
```

The output logs'/checkpoints' location is ```logs/Dual-MViT-B/``` .


# Test
Some test codes are provided here, [test by mulit-process datasetloader] or [a predictor one by one(much slower, 6x time on my hardware...)]
## Test by datasetloader

### Prepare the dataset dir
``` bash
path/to/dataset
|-- test.csv
|-- testset
|   |-- 02cbac6f98b8a9a33bf8206b88b11417.mp4
|   |-- 52cbac6f98b8a92d3bf8206b88b11417.mp4
...
```
As same as the train stage, the test csv looks like below lines:
``` sql
video_name,target
4663658dbe76be9de4d616b573546a0c.mp4,0
8fedbd1856eb89741d7488247f05d7e8.mp4,0
5a5275506a58ed0b27e3288c042a6d9e.mp4,0
7d27e58b3a8273a7ee5f9efd5440fd4f.mp4,0
...
```

### Prepare checkpoint
Download the BaiduDisk's checkpoint (scroll down the the download link) and put into the project root. Asume the path is ```path/to/checkpoint```

### Run script
``` bash
python train_video_demo.py path/to/dataset_root -m test -c path/to/checkpoint
```

## Test by predictor (One by One Example)
This is an ```example.py``` (this file is under the project root) to show how to use the predictor, the details is in ```predict_one_by_one.py```

``` python
# example.py

from predict_one_by_one import VideoPredictor
from pathlib import Path
import tqdm, os, glob, pandas as pd

# Video Director 
all_video_dir = "path/to/videos"
# Test checkpoint
checkpoint = "path/to/checkpoint"
# Enable FP16 or not
fp_16 = False

# Output result csv file like submit csv
output_csv = "val_submit_final.csv"

# Init the predictor with the checkpoint, disable FP16
predictor = VideoPredictor(checkpoint, fp_16=fp_16)

videos = glob.glob(os.path.join(all_video_dir, "*"))

submit_csv = {
    "video_name": [],
    "y_pred": [],
}
l = len(videos)
for idx, i in tqdm.tqdm(enumerate(videos), leave=False, position=0, total=l):

    # Call predictor to get the fake prob
    res = predictor(i)

    # Save to csv file
    submit_csv["video_name"].append(Path(i).name)
    submit_csv["y_pred"].append(res)
    print(res)
    if idx % 10000 == 0 and idx > 0:
        pd.DataFrame(submit_csv).to_csv(f"val_submit_{idx:06}.csv", index=False)

pd.DataFrame(submit_csv).to_csv(output_csv, index=False)

```
## Check the results
The follow results are test in my local server on public testset, compare the results to confirm the test steps.
> 通过下方缩略文档来检查环境和模型推理是否正确。以下为公开测试集部分数据的本地推理结果。请检查每个输出的误差来确认硬件和软件环境是否会导致较大的精度误差。
``` sql
video_name,y_pred
ebda0cb17c9bc4e9818c9250360b7909.mp4,0.9993643164634703
639954dde40d384708f24213dca21b11.mp4,0.2968801259994507
a8267d3bb58a7f69b996183bb5d9f568.mp4,0.9998409748077391
d3467d19d8b0f791388fa7f50a980f4c.mp4,0.9989484945933024
966308f62a9d14880ef1105b145c7ca7.mp4,0.9989816546440125
2ca207880bd45c645dbf5e566b8e91f3.mp4,0.5712887048721313
cb1efe6f50549d684e10ca95b07735fb.mp4,0.7863392929236094
9979928af98e667907667ae6502f0f16.mp4,0.9999123811721802
```


## Test checkpoint
best checkpoint on public testset, ROC-AUC: 0.722 \
https://pan.baidu.com/s/1vl3wWNbzuVp737zzwSqk1w?pwd=6xme

## Contact
If any questions, please contact me: \
nowandfuture \
1183424701@qq.com