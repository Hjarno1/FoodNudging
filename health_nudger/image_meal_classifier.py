import os, torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from typing import Tuple, List

class ImageMealClassifier:
    def __init__(self):
        # 1) instantiate with the new API
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()

        # 2) preprocessing (shorter side →256, center crop →224)
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
              mean=[0.485,0.456,0.406],
              std =[0.229,0.224,0.225]
            )
        ])

        # 3) load labels from the same folder as this file
        here = os.path.dirname(os.path.abspath(__file__))
        labels_path = os.path.join(here, "imagenet_classes.txt")
        with open(labels_path, "r") as f:
            self.imagenet_labels = [l.strip() for l in f]

    def classify_meal_from_image(self, image_path: str) -> Tuple[str, List[str]]:
        img = Image.open(image_path).convert("RGB")
        inp = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(inp)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        # pick top‐1
        top1_prob, top1_idx = torch.max(probs, 0)
        label = self.imagenet_labels[top1_idx]
        low = label.lower()

        # 4) map to a “meal name” + ingredient list
        if "pizza" in low:
            return "Pizza", ["cheese", "tomato sauce", "flour"]
        elif "burger" in low:
            return "Burger", ["beef patty", "bun", "lettuce"]
        # … your other heuristics …
        else:
            return label, ["lettuce", "tomato"]
