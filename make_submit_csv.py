# video_name,y_pred
import pandas as pd

def convert2submition(val_csv):
    df = pd.read_csv(val_csv)
    df["video_name"] = df["id"]
    df["y_pred"] = df["pred"]
    
    df[["video_name", "y_pred"]].to_csv("val_submit.csv", index=False)

def convert2submition_image(results):
    df = pd.read_csv(results)
    df["img_name"] = df["file_info"]
    df["y_pred"] = df["pred_logits"].apply(lambda x: float(x.split(",")[1][:-1]))
    
    df[["img_name", "y_pred"]].to_csv("val_submit_image.csv", index=False)

if __name__ == "__main__":
    convert2submition("/nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/_merged_av_res_testset.csv")
    # convert2submition_image("/nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/work_dir/EfficientNet_B0_NPR_GRAD_deepfake_c2_002_test/results.csv")