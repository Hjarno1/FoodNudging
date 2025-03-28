import re
from typing import List, Tuple

class TextIngredientDetector:

    # A very simple list of known ingredients for pattern matching
    KNOWN_INGREDIENTS = [
        'white bread', 'pepperoni', 'mayo', 'full-fat cheese', 
        'lettuce', 'tomato', 'chicken', 'whole wheat bread',
        'turkey pepperoni', 'greek yogurt', 'reduced-fat cheese'
    ]

    @staticmethod
    def detect_ingredients_from_text(text: str) -> List[str]:
        text_lower = text.lower()
        found_ingredients = []
        
        # Simple approach: check if each known ingredient appears in the text
        for ingr in TextIngredientDetector.KNOWN_INGREDIENTS:
            if ingr in text_lower:
                found_ingredients.append(ingr)
        
        return found_ingredients


class ImageMealClassifier:
    """
    Mock class for image-based meal detection.
    Here we will use a ML model to classify the meal type and ingredients.
    """

    def classify_meal_from_image(self, image_path: str) -> Tuple[str, List[str]]:
        """
        Returns a mock classification (meal type, list of mock ingredients)
        purely for demonstration.
        """
        # In a real scenario, you'd load the image, run it through a model like
        # a pretrained ResNet, YOLO, etc., and detect meal type + ingredients.
        
        # For demonstration, let's just return a pretend result:
        # e.g. "pizza" with "pepperoni" and "full-fat cheese"
        # You could randomize or base it on the file name, etc.
        # We'll do a simple example:
        if "pizza" in image_path.lower():
            return ("Pepperoni Pizza", ["pepperoni", "full-fat cheese"])
        elif "burger" in image_path.lower():
            return ("Cheeseburger", ["full-fat cheese", "mayo", "white bread"])
        else:
            # default mock
            return ("Salad", ["lettuce", "tomato", "chicken"])
