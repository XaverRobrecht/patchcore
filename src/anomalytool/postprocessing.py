from matplotlib import pyplot as plt
from PIL import Image,ImageDraw
import numpy as np
from skimage import measure


def generateHeatmap(pixel_scores, 
                    path_to_iamge_file, 
                    th=None, 
                    colormap_name = "inferno", 
                    border_color = "orange", 
                    crop_tuple=None, 
                    target_size=None, 
                    ):
    """ 
    Generates a heatmap from pixel scores and overlays it on the original image.
    The heatmap is created using a specified colormap and blended with the original image.
    Contours of the heatmap are highlighted with a specified border color.
    Args:
        pixel_scores (np.ndarray): 2D array of pixel scores.
        path_to_iamge_file (str): Path to the original image file.
        th (float, optional): Threshold for contour detection. Defaults to None.
        colormap_name (str, optional): Name of the colormap to use. Defaults to "inferno".
        border_color (str, optional): Color of the contour border. Defaults to "orange".
        crop_tuple (tuple, optional): Tuple specifying the crop area (left, upper, right, lower). Defaults to None.
        target_size (tuple, optional): Target size for resizing the image (width, height). Defaults to None.
    Returns:
        PIL.Image: The blended image with heatmap overlay and highlighted contours.
    """
    cmap = plt.get_cmap(colormap_name)
    image = Image.open(path_to_iamge_file).convert("RGBA")
    if crop_tuple is not None:
        image = image.crop(crop_tuple)
    if target_size is not None:
        image = image.resize(target_size)

    #apply threshold and normalize pixel scores

    contours = []
    if th is not None:
        contours = measure.find_contours(pixel_scores, th)
    pixel_scores = np.maximum(pixel_scores-th,0)
    if pixel_scores.max() > 0:
        pixel_scores = pixel_scores/pixel_scores.max()

    #create a heatmap
    heatmap = Image.fromarray(np.uint8(cmap(pixel_scores)*255)).convert("RGBA")
    heatmap=heatmap.resize(image.size)

    #blend heatmap and image
    out = Image.blend(image,heatmap,.7).convert("RGB")
    draw = ImageDraw.Draw(out)

    #highlight contour
    old_height, old_width = pixel_scores.shape
    new_width,new_height = out.size
    kx =  new_width/(1.0*old_width)
    ky =  new_height/(1.0*old_height)
    for contour in contours:
        points = [(int(kx*x),int(ky*y)) for y,x in contour]
        draw.polygon(points,outline = border_color,width=2)
    return out



