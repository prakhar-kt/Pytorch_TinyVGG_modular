import torch
import torchvision
from torchvision import transforms
# import matplotlib.pyplot as plt
from pathlib import Path

import model_builder

import argparse

from typing import Tuple, Dict, List

from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Get the model name")

parser.add_argument("--model_name",
                    type=str,
                    default="tiny_vgg.pth",
                    help="Name of the model to be loaded.")

parser.add_argument("--image_path",
                    type=str,
                    help="Path to the image to be predicted.")

args = parser.parse_args()

MODEL_NAME = args.model_name
IMG_PATH = args.image_path
MODEL_DIRECTORY = "model_checkpoints"

class_names = ["pizza", "steak", "sushi"]

def load_model(model_dir: str = MODEL_DIRECTORY,
               model_name: str = MODEL_NAME):
    
    """
    Loads a model from a directory.

    Args:
            
            model_dir: str
                The directory where the model is saved.
    
            model_name: str
                The name of the model to be loaded.

    Returns:
                
            A torch.nn.Module model.
    
        
    """

    model_path = Path(model_dir) / model_name

    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=10,
        output_shape=3

    ).to(device)

    model.load_state_dict(torch.load(model_path))

    return model



def predict(image_path: str = IMG_PATH,
            model_name:str = MODEL_NAME,
            device: torch.device = device):
    
    """
    Predicts the class of an image.

    Args:
            
            image_path: str
                The path to the image to be predicted.

            model_name: str
                The name of the model to be loaded.

            device: torch.device
                The device to run the model on.

    Returns:    
                
                A tuple of the predicted class and the probability of the prediction.
        
            
        """ 
    
    model = load_model(model_name=model_name)

    image = torchvision.io.read_image(image_path)

    image = image / 255.

    transform = torchvision.transforms.Resize((64,64))

    image = transform(image)

    # predict the class of the image

    model.eval()

    with torch.inference_mode():

        image = image.to(device)

        image = image.unsqueeze(0)

        pred_logits = model(image)

        pred_probs = torch.softmax(pred_logits, dim=1)

        pred_label = torch.argmax(pred_probs, dim=1)

        pred_class = class_names[pred_label]

    print(f"Predicted class: {pred_class}")

if __name__ == "__main__":

    predict()