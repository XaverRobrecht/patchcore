from .models import OriginalAggregator, PatchCore,Scorer,FeatureExtractor
from .preprocessing import ImageLoader
from torchvision.models import resnet50,wide_resnet50_2, Wide_ResNet50_2_Weights,ResNet50_Weights
from tqdm import tqdm
import torch
from random import randint

def createWideResnet50Backbone(layer_nums):
    layer_nums.sort()
    layers = [f"layer{i}" for i in range(1,5)]
    if max(layer_nums)>4 or min(layer_nums)<1:
        raise ValueError("layer numbers must be in range [1,4]")
    
    weights = Wide_ResNet50_2_Weights.DEFAULT
    complete_model = wide_resnet50_2(weights=weights)
    layers = [layers[i-1] for i in layer_nums]
    backbone =  FeatureExtractor(   
                                    model = complete_model,
                                    return_nodes=layers, 
                                )
    return backbone



def subsample_vectorset(vector_set, n_sample,device=None,target_dim = None):
    """greedy subsamples a vector set to n_sample vectors.
    The vectors are selected such that the distance to the already selected vectors is maximized.
    This is done by selecting the first vector randomly and then iteratively selecting the vector with the maximum distance to the already selected vectors.
    The distance is calculated as the squared euclidean distance.
    If the target_dim is specified, the vectors are first projected to the target dimension using random projection.

    Args:
        vector_set (torch tensor): the vector set to subsample, must be of shape (n_vectors, vector_dim).
        n_sample (int): the number of vectors to subsample from the vector set.
        device (torch.device, optional): device to run computations on. Defaults to choosing gpu if available, else chooses cpu.
        target_dim (int, optional): if given, uses a randomly projected version of vectors_set to do the sampling. Defaults to None.

    Returns:
        torch tensor: the subsampled vector set of shape (n_sample, vector_set.shape[1]).
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device="cpu"
    
    #handle subsampling
    vs = vector_set
    if target_dim is not None:
        if target_dim <=0 :
            eps = 1.0
            bank_size = torch.as_tensor(vector_set.shape[0],dtype = torch.float32,device=device)
            target_dim = (4*torch.log(bank_size)/(eps**2/2.0 - eps**3/3.0)).to(dtype=torch.int64)
        vs = random_projection(vs,target_dim,device)
    

    idx = [randint(0,int(vs.shape[0]))]
    dists = ((vs[idx[0]]-vs)**2).sum(1).to(device)
    
    #constant term for distance computation
    cT = torch.pow(vs,2).sum(1).unsqueeze(0)
    
    #aas torch.compile does not work on windows with gpu, for now its jit.trace
    gC = torch.jit.trace(_getCandidate,(vs[idx[0]],dists,vs,cT))
    for i in tqdm(range(1,n_sample),"subsampling memory bank",leave=False):
        current_idx = torch.argmax(dists)
        dists,= gC(vs[current_idx], dists,vs,cT)
        idx.append(current_idx)
    idx = torch.as_tensor(idx).to(device)
    return vector_set[idx].to("cpu")

@torch.no_grad
def _getCandidate(fv, dists,vs,cT):
    fv=fv.unsqueeze(0)
    dists2 = torch.pow(fv,2).sum(1).unsqueeze(1)+cT + -2*(fv.matmul(vs.T))
    dists = torch.minimum(dists,dists2)
    return dists

@torch.no_grad
def random_projection(matrix,target_dimension,device = None):
    """performs a random projection of the given matrix to the target dimension.
    The projection is created by sampling from a normal distribution with standard deviation 1/sqrt(n) where n is the dimensionality of the target space.
    If the device is not specified, it will choose the first available gpu or cpu."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device="cpu"
    std = torch.zeros((matrix.shape[1],target_dimension),dtype=torch.float32)
    std += 1.0/target_dimension
    std = torch.sqrt(std)
    projection = torch.normal(std=std,mean=0*std).to(device)
    return matrix.to(device).matmul(projection)

def train_patchcore(training_files, 
                    backbone = createWideResnet50Backbone([2,3]), 
                    aggregator= OriginalAggregator(3,1024,1024),
                    n_percent=25, 
                    b=10, 
                    sigma=4, 
                    crop_tuple=None, 
                    target_dim=None,
                    mask=None, 
                    batch_size=1, 
                    device="cpu"):
    """Utility function to train a patchcore anomaly detector

    Args:
        training_files (list(str)): list of strings containing the path to the training images
        backbone (torch.nn.module, optional): Model used to extract fecture vectors from image. Defaults to ideResnet50 with ooutput layers 2,3.
        aggregator (torch.nn.module, optional): Module that perfomrs the aggregation on the extracted features. Defaults to OriginalAggregator(3,1024,1024), which is what the paper bechmark used.
        n_percent (int, optional): percentage of feature vectors in memory bank to keep. Defaults to 25.
        b (int, optional): number of nearest neighbours used when scoring images. Defaults to 10.
        sigma (int, optional): standard deviation of gaussian kernel used to smooth background outside masked region. Defaults to 4.
        crop_tuple (Tuple[int,int,int,int], optional): tuple used to crop pillow image with, corresponds to (left,upper,right,lower). Defaults to None/no cropping.
        target_dim (int, optional): Specifies if a random projection is used during subsampling. Defaults to None(=no random projection), if <=0  dimension will be chosen automatically.
        mask (list[list[tuple]], optional): polygons used to crop a part of a certain image. Defaults to None(=no cropping).
        batch_size (int, optional): Batch size used to extract memory bank. Defaults to 1.
        device (str, optional): torch device to run models on. Defaults to "cpu".

    Raises:
        ValueError

    Returns:
        torch.nn.module: trained patchcore classifier
    """
    
    image_loader = ImageLoader(crop_tuple=crop_tuple,
                                      sigma=sigma,
                                      mask=mask,
                                      device=device)
    
    feature_extractor = backbone
    feature_extractor.eval()
    feature_extractor.to(device)
    aggregator.to(device)
    
    # Load and preprocess images batchwise
    features = []
    batched_files = [training_files[i:i + batch_size] for i in range(0, len(training_files), batch_size)]
    
    with tqdm(total=len(training_files),desc="extracting memory bank", leave=False) as pbar:   
        for files in batched_files:
            tensor = image_loader.load_images_as_tensor(files)
            tensor = aggregator(feature_extractor(tensor))
            feature_vectors = tensor.reshape((-1,tensor.shape[-1]))
            features.append(feature_vectors)
            pbar.update(len(files))
    features = torch.cat(features, dim=0)
    
    # Subsample vectorset
    n_sample = int(n_percent/100.0 * features.shape[0])
   

    if n_percent >=100:
        mb = features.to("cpu")
    else:
        if n_sample > features.shape[0]:
            raise ValueError("n_percent is too high, more samples requested than available in the dataset.")
        if n_sample <= 0:
            raise ValueError("n_percent must be greater than 0.")
        mb = subsample_vectorset(features, n_sample,target_dim=target_dim)
        
    
    scorer = Scorer(b=b, memory_bank=mb)
    scorer.eval()
    scorer.to(device)
    
    patchcore = PatchCore(
                        scorer=scorer,
                        feature_extractor=feature_extractor,
                        sigma=sigma,
                        aggregator = aggregator
                        )
    
    patchcore.eval()
    patchcore.to(device)


    return patchcore,image_loader
    