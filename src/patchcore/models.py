import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torch.nn.functional as F
import math


class FeatureExtractor(nn.Module):
    """
    Module for extracting and aggregating features from intermediate layers of a model.

    Args:
        return_nodes (list): list of node names to extract from the model.
        model (nn.Module): Backbone model to extract features from.
        mean (list, optional): Mean for normalization. Defaults to ImageNet mean.
        std (list, optional): Standard deviation for normalization. Defaults to ImageNet std.
        input_size (tuple, optional): Input image size (H, W). Defaults to (224, 224).
    """
    def __init__(
        self,
        return_nodes,
        model,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        input_size=(224, 224)
    ):
        """
        Initializes the FeatureExtractor.

        Args:
            return_nodes (dict): Mapping of node names to extract from the model.
            model (nn.Module): Backbone model to extract features from.
            mean (list, optional): Mean for normalization. Defaults to ImageNet mean.
            std (list, optional): Standard deviation for normalization. Defaults to ImageNet std.
        """
        super(FeatureExtractor, self).__init__()
        self.model = create_feature_extractor(
            model=model, return_nodes=return_nodes
        )

        self.input_size = input_size
        mean = torch.as_tensor(mean).to(torch.float32)
        std = torch.as_tensor(std).to(torch.float32)
        self.mean = torch.nn.Parameter(mean.reshape((1, 3, 1, 1)))
        self.std = torch.nn.Parameter(std.reshape((1, 3, 1, 1)))
        self.mean.requires_grad = False
        self.std.requires_grad = False

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature extraction.

        Args:
            input_tensor (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Aggregated feature tensor.
        """
        # scale and normalize w.r.t. imagenet
        input_tensor = input_tensor.to(torch.float32)
        input_tensor = input_tensor / 255.0
        input_tensor = F.interpolate(input_tensor, size=self.input_size,mode="bilinear")
        input_tensor = (input_tensor - self.mean) / self.std
        model_outputs = self.model(input_tensor).values()
        return list(model_outputs)
        

class OriginalAggregator(torch.nn.Module):
    def __init__(self,patch_size,pretrain_embed_dimension=1024,target_embed_dimension=1024):
        super(OriginalAggregator, self).__init__()
        self.pretrain_embed_dimension = pretrain_embed_dimension
        self.target_embed_dimension = target_embed_dimension
        self.patch_size = patch_size
    
    @torch.no_grad
    def forward(self,backbone_outputs):
        fmp_height = max([ft.shape[2] for ft in backbone_outputs])
        fmp_width = max([ft.shape[3] for ft in backbone_outputs])
        scaled_features = []
        for fmp in backbone_outputs:
            batch,channels,_,_ = fmp.shape
            padding = (self.patch_size-1)//2
            patched_features = F.unfold(fmp,(self.patch_size,self.patch_size),padding=(padding,padding))
            h = w = torch.sqrt(torch.as_tensor(patched_features.shape[-1])).to(torch.int64)
            patched_features = patched_features.reshape((batch,-1,h,w))
            patched_features = F.interpolate(patched_features,
                                             (fmp_height,fmp_width),
                                             mode="bilinear",
                                             align_corners=False,)
            patched_features = patched_features.reshape((batch,channels,-1,fmp_height,fmp_width))
            patched_features = torch.permute(patched_features,(0,1,3,4,2))
            scaled_features.append(patched_features)

        scaled_features = torch.cat(scaled_features,dim=1)
        batch,channels,height,width,patch = scaled_features.shape 
        scaled_features = scaled_features.reshape((-1,height*width*patch))

        aggregated_features = torch.adaptive_avg_pool1d(scaled_features,self.pretrain_embed_dimension).reshape((batch,channels,self.pretrain_embed_dimension))
        aggregated_features = torch.adaptive_avg_pool1d(aggregated_features,self.target_embed_dimension)
        h = w = torch.sqrt(torch.as_tensor(self.target_embed_dimension)).to(torch.int64)
        aggregated_features = aggregated_features.reshape((batch,channels,h,w))
        features = torch.permute(aggregated_features, (0, 2, 3,1))
        return features
    
class AverageAggregator(torch.nn.Module):
    def __init__(self,patch_size):
        super(AverageAggregator, self).__init__()
        self.patch_size = patch_size
    
    @torch.no_grad
    def forward(self,backbone_outputs):
        fmp_height = max([ft.shape[2] for ft in backbone_outputs])
        fmp_width = max([ft.shape[3] for ft in backbone_outputs])
        scaled_features = []
        for fmp in backbone_outputs:
            batch,channels,_,_ = fmp.shape
            padding = (self.patch_size-1)//2
            patched_features = F.unfold(fmp,(self.patch_size,self.patch_size),padding=(padding,padding))
            h = w = torch.sqrt(torch.as_tensor(patched_features.shape[-1])).to(torch.int64)
            patched_features = patched_features.reshape((batch,-1,h,w))
            patched_features = F.interpolate(patched_features,
                                             (fmp_height,fmp_width),
                                             mode="bilinear",
                                             align_corners=False,)
            patched_features = patched_features.reshape((batch,channels,-1,fmp_height,fmp_width))
            patched_features = torch.permute(patched_features,(0,1,3,4,2))
            scaled_features.append(patched_features)

        scaled_features = torch.cat(scaled_features,dim=1)
        #batch,channels,height,width,patch = scaled_features.shape 
        aggregated_features = scaled_features.mean(dim=4)
        
        #channels last
        features = torch.permute(aggregated_features, (0, 2, 3,1))
        return features
    
class OriginalScorer(torch.nn.Module):
    """
    Computes anomaly scores for feature maps using a memory bank.

    Args:
        memory_bank (torch.Tensor): Memory bank of feature vectors (N, C).
        b (int): Number of nearest neighbors to use for scoring.
        scale_width (int): Output width for pixel scores.
        scale_height (int): Output height for pixel scores.
    """
    def __init__(self, memory_bank, b):
        """
        Initializes the Scorer.

        Args:
            memory_bank (torch.Tensor): Memory bank of feature vectors (N, C).
            b (int): Number of nearest neighbors to use for scoring.
            scale_width (int): Output width for pixel scores.
            scale_height (int): Output height for pixel scores.
        """
        super(Scorer, self).__init__()
        self.mb = torch.nn.Parameter(memory_bank,requires_grad = False)
        self.b = torch.nn.Parameter(torch.as_tensor(b).to(torch.int64),requires_grad = False)

    @torch.no_grad()
    def forward(self, feature_batch):
        """
        Computes pixel-wise and image-level anomaly scores.

        Args:
            feature_batch (torch.Tensor): Feature tensor of shape (B, H, W, C).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pixel-wise scores of shape (B, 1, H, W) and image-level scores of shape (B,).
        """
        batch, height, width, channels = feature_batch.shape
        device = next(self.parameters()).device
        featureVectors = torch.reshape(feature_batch, (batch * height * width, channels))

        # Calculate distances as pixel scores
        squared_dists = torch.pow(featureVectors,2).sum(1).unsqueeze(1) + torch.pow(self.mb,2).sum(1).unsqueeze(0)-2*(featureVectors.matmul(self.mb.T))
        pixel_scores = squared_dists.min(dim=1)[0]
        pixel_scores = pixel_scores.reshape((batch, height * width))
        pixel_scores = torch.sqrt(pixel_scores)

        # Select most anomalous pixel score per image
        idx = torch.argmax(pixel_scores, dim=1)
        flattend_idx = idx + torch.arange(batch,device=device) * height * width

        # Get the b closest elements in mb for the selected vectors
        selected_vectors = featureVectors[flattend_idx, :]
        dists_to_mb = torch.pow(selected_vectors.unsqueeze(1) - self.mb.unsqueeze(0), 2).sum(dim=2)
        b_closest_idx = torch.topk(dists_to_mb, k=self.b, dim=1, largest=False, sorted=True)[1]

        # Calculate the distances
        selected_dists = torch.zeros((batch, self.b),device=device)
        for i, indices in enumerate(b_closest_idx):
            fv = featureVectors[flattend_idx[i], :]
            fv = fv.reshape((1, 1, channels))
            mb = self.mb[indices, :]
            mb = mb.reshape((1, self.b, channels))
            selected_dists[i] = (torch.pow(fv - mb, 2).sum(dim=2)).flatten()
        selected_dists = torch.sqrt(selected_dists)

        # Do the reweighting
        image_scores = selected_dists[:, 0]
        if selected_dists.shape[1] >1:
            image_scores = image_scores * (1 - F.softmax(selected_dists, dim=1)[:, 0])

        #fix shape
        pixel_scores = pixel_scores.reshape(batch, 1, height, width)
        return pixel_scores, image_scores
    

class Scorer(torch.nn.Module):
    """
    Computes anomaly scores for feature maps using a memory bank.

    Args:
        memory_bank (torch.Tensor): Memory bank of feature vectors (N, C).
        b (int): Number of nearest neighbors to use for scoring.
        scale_width (int): Output width for pixel scores.
        scale_height (int): Output height for pixel scores.
    """
    def __init__(self, memory_bank, b):
        """
        Initializes the Scorer.

        Args:
            memory_bank (torch.Tensor): Memory bank of feature vectors (N, C).
            b (int): Number of nearest neighbors to use for scoring.
            scale_width (int): Output width for pixel scores.
            scale_height (int): Output height for pixel scores.
        """
        super(Scorer, self).__init__()
        self.mb = torch.nn.Parameter(memory_bank,requires_grad = False)
        self.b = torch.nn.Parameter(torch.as_tensor(b).to(torch.int64),requires_grad = False)

    @torch.no_grad()
    def forward(self, feature_batch):
        """
        Computes pixel-wise and image-level anomaly scores.

        Args:
            feature_batch (torch.Tensor): Feature tensor of shape (B, H, W, C).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pixel-wise scores of shape (B, 1, H, W) and image-level scores of shape (B,).
        """
        batch, height, width, channels = feature_batch.shape
        device = next(self.parameters()).device
        featureVectors = torch.reshape(feature_batch, (batch * height * width, channels))

        # Calculate distances as pixel scores
        squared_dists = torch.pow(featureVectors,2).sum(1).unsqueeze(1) + torch.pow(self.mb,2).sum(1).unsqueeze(0)-2*(featureVectors.matmul(self.mb.T))
        top_k_dists_squared = torch.topk(squared_dists, k=self.b, dim=1, largest=False, sorted=True)[0]
        pixel_scores = top_k_dists_squared[:,0].reshape((batch, height * width))
        pixel_scores = torch.sqrt(pixel_scores)
        max_idx = pixel_scores.argmax(dim=1)
        batch_idx = torch.arange(batch).to(device)
        
        #gather closest indices to calculate sotmax
        softmax_args = torch.sqrt(top_k_dists_squared.reshape((batch, height * width,self.b))[batch_idx,max_idx,:])

        image_scores =  softmax_args[:, 0] 
        if softmax_args.shape[1]>1:
            image_scores =  image_scores*(1 - F.softmax(softmax_args, dim=1)[:, 0])
        pixel_scores = pixel_scores.reshape((batch, 1,height , width))
        return pixel_scores,image_scores

class PatchCore(torch.nn.Module):
    """
    PatchCore anomaly detection model.

    Args:
        scorer (nn.Module): Scorer module for computing anomaly scores.
        feature_extractor (nn.Module): Feature extractor module.
        sigma (float): Standard deviation for Gaussian blur.
    """
    def __init__(self, feature_extractor, aggregator, scorer, sigma):
        """
        Initializes the PatchCore model.

        Args:
            scorer (nn.Module): Scorer module for computing anomaly scores.
            feature_extractor (nn.Module): Feature extractor module.
            sigma (float): Standard deviation for Gaussian blur.
        """
        super(PatchCore, self).__init__()
        self.scorer = scorer
        self.feature_extractor = feature_extractor
        self.aggregator=aggregator
        filter_size = int(round(4 * sigma))
        kernel = torch.zeros((filter_size, filter_size))
        center = (filter_size-1) / 2.0
        for i in range(filter_size):
            for j in range(filter_size):
                r2 = (i - center) ** 2 + (j - center) ** 2
                value = math.exp(-r2 / (2 * sigma ** 2)) / (2 * math.pi * sigma ** 2)
                kernel[i, j] = value
        self.blur = torch.nn.Conv2d(1, 1, filter_size, stride=1, padding="same", padding_mode="reflect", bias=False)
        self.blur.weight = torch.nn.Parameter( kernel.reshape(self.blur.weight.shape),requires_grad = False)

    def set_b(self,b):
        self.scorer.b = torch.nn.Parameter(torch.as_tensor(b).to(torch.int64),requires_grad = False)

    @torch.no_grad()
    def forward(self, input_tensor):
        """
        Forward pass for PatchCore.

        Args:
            input_tensor (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pixel-wise anomaly scores and image-level anomaly scores.
        """
        # Ensure batch dimension exists
        if len(input_tensor.shape) < 4:
            input_tensor = input_tensor.unsqueeze(0)

        # Extract features
        features = self.feature_extractor(input_tensor)
        
        #Aggregate
        patch_features = self.aggregator(features)

        # Return scores
        pixel_scores, image_scores = self.scorer(patch_features)
        pixel_scores = F.interpolate(pixel_scores, size=self.feature_extractor.input_size, mode="bilinear")
        pixel_scores = self.blur(pixel_scores)
        pixel_scores = pixel_scores.squeeze(1)
        return image_scores,pixel_scores