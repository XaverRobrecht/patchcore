from .models import PatchCore,Scorer,FeatureExtractor
from .preprocessing import ImageLoader
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from tqdm import tqdm
import torch

def createWideResnet50Backbone(layer_nums,pooling_size):
    """creates a wide resnet50 backbone with the specified layer numbers and pooling size.
    The layer numbers must be in the range [1,4] and specify which resolution blocks to extract features from.
    The pooling size specifies the pooling kernels size vie kernel_size=(pooling_size,pooling_size).

    Args:
        layer_nums (list): specifies which layers to extract features from.
        pooling_size (int): pooling kernel size, must be an integer.

    Raises:
        ValueError: raises an error when the layer numbers are not in the range [1,4].

    Returns:
        torch.nn.module: the feature extractor containing the truncated wide resnet 50 as well as a pooling layer and common input transorms(cropping is omitted).
    """
    layer_nums.sort()
    layers = [f"layer{i}" for i in range(1,5)]
    if max(layer_nums)>4 or min(layer_nums)<1:
        raise ValueError("layer numbers must be in range [1,4]")
    
    weights = Wide_ResNet50_2_Weights.DEFAULT
    complete_model = wide_resnet50_2(weights=weights)
    layers = [layers[i-1] for i in layer_nums]
    backbone =  FeatureExtractor(   
                                    model = complete_model,
                                    input_size = (224,224),
                                    return_nodes=layers, 
                                    pooling_size=pooling_size
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
    
    vs = vector_set
    if target_dim is not None:
        vs = random_projection(vs,target_dim,device)
    idx = torch.zeros(n_sample,dtype = torch.int64).to(device)
    idx[0] = torch.randint(0,vs.shape[0],(1,)).to(device)
    dists = ((vs[idx[0]]-vs)**2).sum(1).to(device)
    for i in tqdm(range(1,n_sample),"subsampling memory bank",leave=False):
        dists,idx[i] = _getCandidate(vs, dists)
    return vector_set[idx].to("cpu")

@torch.compile()
def _getCandidate(bank, dists):
    idx = torch.argmax(dists)
    dists2 = ((bank[idx]-bank)**2).sum(1)
    dists = torch.minimum(dists,dists2)
    return dists,torch.argmax(dists)

@torch.no_grad
def random_projection(matrix,target_dimension,device = None):
    """performs a random projection of the given matrix to the target dimension.
    The projection is created by sampling from a normal distribution with standard deviation 1/sqrt(n) where n is the number of vectors in the matrix.
    If the device is not specified, it will choose the first available gpu or cpu."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device="cpu"
    std = torch.zeros((matrix.shape[1],target_dimension),dtype=torch.float32)
    std += 1.0/matrix.shape[0]
    std = torch.sqrt(std)
    projection = torch.normal(std=std,mean=0*std).to(device)
    return torch.matmul(matrix,projection)

def train_patchcore(training_files, 
                    backbone, 
                    n_percent=25, 
                    b=10, 
                    sigma=4, 
                    crop_tuple=None, 
                    target_size=None, 
                    mask=None, 
                    batch_size=1, 
                    device="cpu"):
    """Generates a PatchCore model from the given training files and backbone.
    The training files are loaded and preprocessed using the ImageLoader.
    The feature extractor is created from the backbone and the images are extracted in batches.
    The features are then subsampled to n_percent of the total number of features.
    A Scorer is created from the subsampled features and the PatchCore model is created from the Scorer and the feature extractor.
    The model is then returned along with the ImageLoader for further use.

    Args:
        training_files (list): list of file paths to the training images.
        backbone (torch.nn.module): the backbone model to use for feature extraction.
        n_percent (int, optional): percentage of the extracted memory bank to keep. Defaults to 25.
        b (int, optional): amount of nearest neighbours to consider while scoring. Should usually be in [2,20]. Defaults to 10.
        sigma (int, optional): Standard deviation of Gaussian Filter used to smooth pixel scores. Defaults to 4.
        crop_tuple (list, optional): List or tuple passed on to PIL.Image.crop. Should contain the crop pixel coordinates (left, upper,right,lower). Defaults to None(no cropping).
        target_size (int, optional): specifies the target dimension of the random projection used in subsampling. If None, no random projection is used. Defaults to None.
        mask ((list[list[tuple]]), optional): list of polygons used for applying a mask after loading the image. Defaults to None.
        batch_size (int, optional): batchsize to load images in. Defaults to 1.
        device (str, optional): specifies which device to run models and sampling on. Defaults to "cpu".

    Raises:
        ValueError: Signals illegal n_sample values

    Returns:
        torch.nn.module: the trained PatchCore model.
        ImageLoader: the ImageLoader used to load and preprocess the images.
    """
    
    image_loader = ImageLoader(crop_tuple=crop_tuple,
                                      target_size=target_size,
                                      sigma=sigma,
                                      mask=mask,
                                      device=device)
    
    feature_extractor = backbone
    feature_extractor.eval()
    feature_extractor.to(device)
    
    # Load and preprocess images batchwise
    features = []
    batched_files = [training_files[i:i + batch_size] for i in range(0, len(training_files), batch_size)]
    
    with tqdm(total=len(training_files),desc="extracting memory bank", leave=False) as pbar:   
        for files in batched_files:
            tensor = image_loader.load_images_as_tensor(files)
            tensor=feature_extractor(tensor)
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
        mb = subsample_vectorset(features, n_sample)
        
    
    scorer = Scorer(b=b, memory_bank=mb)
    scorer.eval()
    scorer.to(device)
    
    patchcore = PatchCore(scorer=scorer,
                                 feature_extractor=feature_extractor,
                                 sigma=sigma)
    
    patchcore.eval()
    patchcore.to(device)


    return patchcore,image_loader
    