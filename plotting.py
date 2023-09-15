from PIL import Image
import pandas as pd
import numpy as np
from ultralytics.utils.plotting import Annotator, colors
import io

def add_bboxs_on_img(image: Image, predict: pd.DataFrame()) -> Image:
    """
    add a bounding box on the image

    Args:
    image (Image): input image
    predict (pd.DataFrame): predict from model

    Returns:
    Image: image whis bboxs
    """
    # Create an annotator object

    annotator = Annotator(np.array(image), line_width=4, font_size=20, pil=True)
    # sort predict by xmin value
    predict = predict.sort_values(by=['xmin'], ascending=True)
    # iterate over the rows of predict dataframe
    for i, row in predict.iterrows():
        # create the text to be displayed on image
        text = f"{row['name']}: {int(row['confidence']*100)}%"
        # get the bounding box coordinates
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        # add the bounding box and text on the image
        annotator.box_label(bbox
                            ,text
                            ,color=colors(row['class'], 'wall' not in row['name'])                            
                            )
    # convert the annotated image to PIL image
    return Image.fromarray(annotator.result())

def get_image_from_bytes(binary_image: bytes) -> Image:
    """Convert image from bytes to PIL RGB format
    
    **Args:**
        - **binary_image (bytes):** The binary representation of the image
    
    **Returns:**
        - **PIL.Image:** The image in PIL RGB format
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image