# Runtime

## My Software Env
conda env create -f environment.yml

## My GPUs
A100 80G x 2

## Train

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

```

The **csv** file contents looks like:
```python
video_name,target
4663658dbe76be9de4d616b573546a0c.mp4,0
8fedbd1856eb89741d7488247f05d7e8.mp4,0
5a5275506a58ed0b27e3288c042a6d9e.mp4,0
7d27e58b3a8273a7ee5f9efd5440fd4f.mp4,0
```

# Test


## Checkpoint
best checkpoint on public testset, ROC-AUC: 0.722c\
链接: https://pan.baidu.com/s/1vl3wWNbzuVp737zzwSqk1w?pwd=6xme
