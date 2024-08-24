from predict_one_by_one import VideoPredictor
from pathlib import Path
import tqdm, os, glob, pandas as pd

# Video Director 
all_video_dir = "path/to/videos"
all_video_dir = "/nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/data/video/phase2/testset1seen/"
# Test checkpoint
checkpoint = "path/to/checkpoint"
checkpoint = "/nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/epoch=3-step=20166.ckpt"
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
    if idx % 20 == 0 and idx > 0:
        pd.DataFrame(submit_csv).to_csv(f"val_submit_{idx:06}.csv", index=False)
        break
pd.DataFrame(submit_csv).to_csv(output_csv, index=False)