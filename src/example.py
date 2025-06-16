from sklearn.metrics import roc_auc_score
import os
from anomalytool import model_generation
from tqdm import tqdm
from PIL import Image
import numpy as np
from anomalytool.postprocessing import generateHeatmap

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

for perc in [1]:
    with open(f"./results_{perc}.txt","w") as resFile:
        resFile.write("Dataset \t  Score \n")

    # Assign directory
    for dataset in os.listdir(mvtec_dir):
        if not os.path.isdir(os.path.join(mvtec_dir,dataset)):
            continue
        directory = os.path.join(mvtec_dir,f"{dataset}/train")
        image_files = getFilesinDir(directory)
        backbone = model_generation.createWideResnet50Backbone([2,3],3)

        #calculate crop shapes to reseble what was used in the paper
        crop_tuple = []
        with Image.open(image_files[0]) as im:
            width,height = im.size
            assert width == height, "image is not square"
            cutoff = width*(1-224.0/256.0)
            cutoff = cutoff/2
            crop_tuple = (cutoff,cutoff,width-cutoff,height-cutoff)

        scorer,loader = model_generation.train_patchcore(image_files, 
                                                         backbone, 
                                                         n_percent=perc, 
                                                         batch_size=batch_size, 
                                                         device=device, 
                                                         b=15, 
                                                         crop_tuple=crop_tuple)
        directory = os.path.join(mvtec_dir,f"{dataset}/test")
        image_files = getFilesinDir(directory)
        
        batched_files = [image_files[i:i + batch_size] 
                         for i in range(0, len(image_files), batch_size) 
        ]
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
        score = roc_auc_score(mask,image_scores)
        
        #save results
        print("\n",f"score in {dataset} is {score}")
        with open(f"./results_{perc}.txt","a") as resFile:
            resFile.write(f"{dataset} \t  {score} \n")

        #save segmentation
        savedir = f"./{dataset}_{perc}"
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        else:
            for fn in os.listdir(savedir):
                os.remove(os.path.join(savedir,fn))
        
        #generate segmentations
        #pixel_th = pixel_scores[np.logical_not(mask)].max()
        #for i,image_file in enumerate(image_files):
        #    org_name = image_file.split("/")
        #    org_name = org_name[-2]+"_"+org_name[-1]
        #    fn = org_name + f"_{image_scores[i]:.3f}.png"
        #    segmentation = generateHeatmap(pixel_scores[i],image_file,pixel_th, crop_tuple=crop_tuple)
        #    segmentation.save(os.path.join(savedir,fn))
        