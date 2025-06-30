from PIL import Image,ImageFilter,ImageDraw
import torch
import numpy as np

class ImageLoader:
    """
    Loads and preprocesses images with optional cropping, blurring, and masking.

    Args:
        crop_tuple (tuple): Tuple (left, upper, right, lower) for cropping the image.
        target_size (tuple): Target size (width, height) for resizing the image.
        sigma (float): Standard deviation for Gaussian blur. If <= 0, no blur is applied.
        mask (list[list[tuple]]): List of polygons, each polygon is a list of (x, y) tuples.
        device (string): Torch device to load tensors into.
    """
    def __init__(self, crop_tuple=None, target_size=None, sigma=0, mask=None,device = "cpu", additional_transform = None):
        """
        Initializes the ImageLoader.

        Args:
            crop_tuple (tuple): Tuple (left, upper, right, lower) for cropping.
            target_size (tuple): Target size (width, height) for resizing.
            sigma (float): Standard deviation for Gaussian blur.
            mask (list[list[tuple]]): List of polygons for masking.
        """
        self.mask = mask
        self.sigma = sigma
        self.crop_tuple = crop_tuple
        self.target_size = target_size
        self.device = device

    def load_image(self, file):
        """
        Loads an image, applies cropping, blurring, and masking.

        Args:
            file (str): Path to the image file.

        Returns:
            PIL.Image: The processed image.
        """
        img = Image.open(file).convert("RGB")

        if self.crop_tuple is not None:
            img = img.crop(self.crop_tuple)
        
        if self.target_size is not None:
            img = img.resize(self.target_size, Image.BILINEAR)

        if self.mask is None or self.mask == []:
            return img

        if self.sigma <= 0:
            blurred = Image.new("RGB", img.size)
        else:
            blurred = img.filter(ImageFilter.GaussianBlur(radius=self.sigma))

        mask = Image.new("L", img.size)
        for polygon in self.mask:
            ImageDraw.Draw(mask).polygon(polygon, fill=255)
        masked_image = Image.composite(img, blurred, mask)

        return masked_image

    def load_image_as_tensor(self, file):
        """
        Loads an image and returns it as a torch tensor.

        Args:
            file (str): Path to the image file.

        Returns:
            torch.Tensor: Image tensor of shape (1, 3, H, W).
        """
        image = self.load_image(file)
        tensor = torch.as_tensor(np.array(image)).unsqueeze(0)
        tensor = torch.permute(tensor, (0, 3, 1, 2))
        tensor.to(self.device)
        return tensor
    
    def load_images_as_tensor(self, files):
        """
        Loads an image batch and returns it as a torch tensor.

        Args:
            files (list): list containing the paths to the image files.

        Returns:
            torch.Tensor: Image tensor of shape (len(files), 3, H, W).
        """
        batch = []
        for file in files:
            image = self.load_image(file)
            tensor = torch.as_tensor(np.array(image)).unsqueeze(0)
            batch.append(tensor)
        
        batch = torch.cat(batch,dim=0).to(self.device)
        batch = torch.permute(batch, (0, 3, 1, 2))
        return batch
    
    def set_device(self,device):
        self.device = device