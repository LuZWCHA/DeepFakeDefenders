import json
import cv2
from PIL import Image
import os

import pandas as pd
import tqdm
from facenet_pytorch import MTCNN
from torchvision.transforms import ToPILImage

def find_faces_in_images():
    folder_path = '/nasdata2/private/lzhao/workspace/kaggle/DeepfakeDatasets/dataset'
    
    mtcnn = MTCNN(keep_all=True)
        
    all_data = "/nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/data/phase1/testset_label.txt"
    images = pd.read_csv(all_data)["img_name"]

    persons_name = []

    for file in tqdm.tqdm(images):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img_path = os.path.join(folder_path, file)
    
            img_obj = Image.open(img_path)
            img_obj = img_obj.convert("RGB")

            boxes, _ = mtcnn.detect(img_obj)
    
            if boxes is not None:
                # print(boxes)
                for box in boxes:
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    
                    W, H = img_obj.size
                    
                    if w*h / (W*H) > 0.25:
                        persons_name.append(img_path)
                        print(f'{img_path} 人脸√')
                        break
    
    print(len(persons_name))
    with open("faces.txt", "w") as f:
        f.writelines([i + "\n" for i in persons_name])
        
def create_face_label_txt(face_txt, dataset_root="/nasdata2/private/lzhao/workspace/kaggle/DeepfakeDatasets/dataset/"):
    df = pd.read_csv(face_txt)
    df["img_name"] = df["img_name"].apply(lambda x: os.path.relpath(x, dataset_root))
    df["target"] = df["img_name"].apply(lambda x: 1 if "1_fake" in x else 0)
    print(df)
    df.to_csv("/nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/data/phase1/testset_face_label.txt", index=False)
    df["target"].hist()
    
if __name__ == "__main__":
    create_face_label_txt("/nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/faces.txt")