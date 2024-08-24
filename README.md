# Runtime

## Software Env
conda env create -f environment.yml

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
```python
video_name,target
4663658dbe76be9de4d616b573546a0c.mp4,0
8fedbd1856eb89741d7488247f05d7e8.mp4,0
5a5275506a58ed0b27e3288c042a6d9e.mp4,0
7d27e58b3a8273a7ee5f9efd5440fd4f.mp4,0
...
```

# Test
Many test ways are provide here, test by mulit-process datasetloader or a predictor one by one(much slower, 6x time on my hardware...)
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
```python
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
This is an ```example.py``` to show how to use the predictor, the details is in ```predict_one_by_one.py```

``` python
# example.py

from predict_one_by_one import VideoPredictor
from pathlib import Path
import tqdm, os, glob, pandass as pd

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
    if idx % 10000 == 0 and idx > 0:
        pd.DataFrame(submit_csv).to_csv(f"val_submit_{idx:06}.csv", index=False)

pd.DataFrame(submit_csv).to_csv(output_csv, index=False)

```

## Test checkpoint
best checkpoint on public testset, ROC-AUC: 0.722 \
https://pan.baidu.com/s/1vl3wWNbzuVp737zzwSqk1w?pwd=6xme

## Contact
If any questions, please contact me: \
nowandfuture \
1183424701@qq.com