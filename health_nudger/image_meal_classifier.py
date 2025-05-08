import os, torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from typing import Tuple, List
from torchvision import models

class ImageMealClassifier:
    def __init__(self):
        # 1) Define the model architecture (same as during training)
        self.model = models.resnet18(weights=None)  # Updated from deprecated 'pretrained'
        num_classes = 4  # Update this to match your number of food classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # 2) Load the trained weights with CPU mapping
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'simple_food_classifier.pth')
            print(f"Attempting to load model from: {model_path}")
            
            # Map the model to CPU explicitly
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Trying alternative paths...")
            
            # Try alternative locations
            alt_paths = [
                'simple_food_classifier.pth',  # Current working directory
                os.path.join(os.path.dirname(current_dir), 'simple_food_classifier.pth'),  # Parent directory
                os.path.join(current_dir, 'data', 'simple_food_classifier.pth')  # data subdirectory
            ]
            
            for alt_path in alt_paths:
                try:
                    if os.path.exists(alt_path):
                        print(f"Found model at: {alt_path}")
                        self.model.load_state_dict(torch.load(alt_path, map_location=torch.device('cpu')))
                        print("Model loaded successfully!")
                        break
                except Exception as alt_e:
                    print(f"Failed to load from {alt_path}: {str(alt_e)}")
            else:
                print("WARNING: Could not load model weights. Using untrained model.")
        
        self.model.eval()  # set to inference mode

        # 3) Image preprocessing transforms (same as during training)
        self.transform = T.Compose([
            T.Resize((224, 224)),     # resize to 224x224 as used in training
            T.ToTensor(),             # convert PIL to PyTorch tensor
            T.Normalize(
              mean=[0.485,0.456,0.406],
              std =[0.229,0.224,0.225]
            )
        ])

        # 4) Define your food class names
        self.food_classes = ['apple_pie', 'hamburger', 'nachos', 'pizza']  # Update with your actual classes
        
        # 5) Define ingredients for each food class (customize as needed)
        self.food_ingredients = {
            'apple_pie': ['apples', 'sugar', 'flour', 'butter'],
            'hamburger': ['beef patty', 'burger buns', 'mayo', 'lettuce'],
            'nachos': ['chips', 'cheese'],
            'pizza': ['white flour', 'full-fat cheese', 'tomato sauce']
        }

    def classify_meal_from_image(self, image_path: str) -> Tuple[str, List[str]]:
        """
        1. Loads and preprocesses the image.
        2. Runs inference using our custom-trained food classifier.
        3. Returns (meal_name, ingredients_list) in the format expected by the rest of the code.
        """
        try:
            # 1) Check if the image file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            # 2) Load and transform the image
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to open or process image: {str(e)}")
                
            input_tensor = self.transform(img).unsqueeze(0)  # shape: [1, 3, 224, 224]

            # 3) Forward pass through the model
            with torch.no_grad():
                outputs = self.model(input_tensor)
            
            # 4) Get the prediction
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_class_idx = torch.max(probabilities, 0)
            
            # 5) Get the predicted food class and confidence
            predicted_class = self.food_classes[top_class_idx.item()]
            confidence = top_prob.item() * 100
            
            # 6) Apply confidence threshold
            if confidence < 40:  # Adjust threshold as needed
                return "Low Confidence Prediction", ["The model is uncertain about this image"]
            
            # 7) Get the ingredients for this food class
            ingredients = self.food_ingredients.get(predicted_class, ["unknown ingredients"])
            
            # 8) Format the meal name with confidence
            meal_name = f"{predicted_class.replace('_', ' ').title()} ({confidence:.1f}% confidence)"
            
            return meal_name, ingredients
            
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
            return "Image Not Found", ["Please provide a valid image path"]
            
        except ValueError as e:
            print(f"Error: {str(e)}")
            return "Invalid Image", ["The provided file could not be processed as an image"]
            
        except IndexError:
            print("Error: Model prediction index out of range")
            return "Classification Error", ["The model failed to classify this image"]
            
        except Exception as e:
            print(f"Unexpected error during classification: {str(e)}")
            return "Classification Failed", ["An unexpected error occurred"]