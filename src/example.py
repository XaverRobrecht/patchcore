from sklearn.metrics import roc_auc_score
import os
from patchcore import model_generation
from patchcore.models import AverageAggregator, OriginalAggregator
from tqdm import tqdm
import numpy as np
from PIL import Image
from random import shuffle

def getFilesinDir(directory):
    image_files = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            path_to_subdir = os.path.join(directory,subdir)
            path_to_file = os.path.join(path_to_subdir, file)
            if os.path.isfile(path_to_file):
                image_files.append(path_to_file)
    return image_files   

mvtec_dir = "/home/xaver/Downloads/mvtec_anomaly_detection/"
batch_size = 1
device="cpu"


subsampling_percentage=1

with open(f"./results_{subsampling_percentage}.txt","w") as resFile:
    resFile.write("Dataset \t  Score \n")

datasets = os.listdir(mvtec_dir)
shuffle(datasets)
for dataset in datasets:
    if not os.path.isdir(os.path.join(mvtec_dir,dataset)):
        continue
    directory = os.path.join(mvtec_dir,f"{dataset}/train")
    image_files = getFilesinDir(directory)
    
    #model creatio

    
    crop_shape=[]
    #cropping as WideResNet50 imagenetv1 proposes
    with Image.open(image_files[0]) as im:
        height,width = im.size
        assert height == width, "images must be square"
        crop = height * (1-224/256.0)
        crop = int(crop/2)
        crop_shape = (crop,crop,width-crop,height-crop)

    scorer,loader = model_generation.train_patchcore(image_files, 
                                                        aggregator=AverageAggregator(patch_size=3),
                                                        n_percent=subsampling_percentage, 
                                                        batch_size=batch_size,
                                                        device=device, 
                                                        crop_tuple=crop_shape,
                                                        target_dim=None)
    
    directory = os.path.join(mvtec_dir,f"{dataset}/test")
    image_files = getFilesinDir(directory)
    
    batched_files = [image_files[i:i + batch_size] 
                        for i in range(0, len(image_files), batch_size) 
    ]
    score = []
    b_vals = [1]
    for b in b_vals:
        scorer.set_b(b)
        scores = [ 
            scorer(loader.load_images_as_tensor(x))  
            for x in tqdm(batched_files,"scoring images",leave=False) 
        ]
        image_scores = [
            x.numpy(force=True)
            for batch in scores
            for x in batch[0]
        ]      
        pixel_scores = np.array([
            x.numpy(force=True)
            for batch in scores
            for x in batch[1]
        ])
        
        mask = ["good" not in x for x in image_files]
        score.append(roc_auc_score(mask,image_scores))
        
    
    idx = np.argmax(score)
    score = score[idx]
    b = b_vals[idx]

    #save results
    print(f"score in {dataset} is {score} for b = {b}")
    with open(f"./results_{subsampling_percentage}.txt","a") as resFile:
        resFile.write(f"score in {dataset} is {score} for b = {b}, p= {3} \n")
