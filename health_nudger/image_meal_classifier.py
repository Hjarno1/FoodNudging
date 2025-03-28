import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from typing import Tuple, List

class ImageMealClassifier:
    """
    Example classifier using a pretrained ResNet on ImageNet.
    This won't be as accurate on food specifically unless you fine-tune
    on a food dataset, but demonstrates the general flow.
    """

    def __init__(self):
        # 1) Load a pretrained model (ResNet18) from TorchVision
        self.model = models.resnet18(pretrained=True)
        self.model.eval()  # set to inference mode

        # 2) Image preprocessing transforms
        self.transform = T.Compose([
            T.Resize(256),            # resize the shorter edge to 256
            T.CenterCrop(224),        # crop out the center 224x224
            T.ToTensor(),             # convert PIL to PyTorch tensor
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 3) Load the ImageNet class names
        with open("imagenet_classes.txt") as f:
            self.imagenet_labels = [line.strip() for line in f.readlines()]

    def classify_meal_from_image(self, image_path: str) -> Tuple[str, List[str]]:
        """
        1. Loads and preprocesses the image.
        2. Runs inference using a pretrained ResNet.
        3. Returns (meal_name, ingredients_list) in the format expected by the rest of the code.

        Note: Because it's still an ImageNet model, we do a best-effort guess.
              We might do extra logic if the predicted label is something "food-like."
        """

        # 1) Load and transform the image
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0)  # shape: [1, 3, 224, 224]

        # 2) Forward pass through the model
        with torch.no_grad():
            outputs = self.model(input_tensor)  # shape: [1, 1000]
        
        # 3) Get the top 5 predictions
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        # 4) Convert the top prediction to a “meal name”
        top1_label = self.imagenet_labels[top5_catid[0]]
        top1_prob = top5_prob[0].item()

        # In a real use-case, you’d have a custom mapping from
        # imagenet_labels -> actual meal categories or ingredients.
        # For now, let’s just pretend that if the top label is “pizza” or “cheeseburger”, etc.
        
        meal_name = top1_label  # e.g. "pizza"
        
        # As for “ingredients”, we might do something naive:
        # For demonstration, let’s do a small mapping. 
        # In real usage, you'd do a food-specific detection or classification pipeline.
        
        recognized_ingredients = []
        # Example of a naive mapping:
        # If top1_label = "pizza", we might guess ["pepperoni", "full-fat cheese"].
        # If top1_label = "cheeseburger", guess ["mayo", "white bread", "full-fat cheese"].
        # Otherwise, default to something like ["lettuce", "tomato"].
        
        lower_label = top1_label.lower()
        if "pizza" in lower_label:
            meal_name = "Pizza"
            recognized_ingredients = ["pepperoni", "full-fat cheese"]
        elif "burger" in lower_label or "cheeseburger" in lower_label:
            meal_name = "Cheeseburger"
            recognized_ingredients = ["white bread", "mayo", "full-fat cheese"]
        elif "sandwich" in lower_label:
            meal_name = "Sandwich"
            recognized_ingredients = ["white bread", "mayo"]
        else:
            # fallback
            meal_name = top1_label
            recognized_ingredients = ["lettuce", "tomato"]

        return meal_name, recognized_ingredients
