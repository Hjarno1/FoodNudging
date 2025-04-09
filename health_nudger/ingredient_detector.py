import re
from typing import List, Tuple

class TextIngredientDetector:
    """Enhanced ingredient detector with better pattern matching and healthy food recognition"""

    # Expanded list of known ingredients (healthy and unhealthy)
    KNOWN_INGREDIENTS = [
        # Unhealthy/common substitution targets
        'white bread', 'pepperoni', 'mayo', 'full-fat cheese', 'ranch dressing', 
        'croutons', 'fried chicken', 'processed lunch meat', 'flavored yogurt',
        'table salt', 'sugar', 'chocolate chips', 'cream', 'granola', 'energy bars',
        
        # Healthy ingredients
        'grilled chicken', 'baked chicken', 'roasted turkey', 'salmon', 'tuna',
        'quinoa', 'brown rice', 'sweet potatoes', 'kale', 'spinach', 'avocado',
        'olive oil', 'nuts', 'seeds', 'berries', 'whole wheat bread', 'greek yogurt'
    ]

    # Healthy ingredients that deserve praise
    HEALTHY_STARS = {
        'grilled chicken': "an excellent lean protein source",
        'salmon': "rich in omega-3 fatty acids",
        'quinoa': "a complete protein with fiber",
        'kale': "packed with vitamins and antioxidants",
        'avocado': "full of healthy monounsaturated fats",
        'olive oil': "heart-healthy fats",
        'berries': "antioxidant-rich fruits",
        'nuts': "great source of healthy fats and protein"
    }

    @staticmethod
    def detect_ingredients_from_text(text: str) -> Tuple[List[str], List[str]]:
        """Returns tuple of (found_ingredients, healthy_ingredients)"""
        text_lower = text.lower()
        found_ingredients = []
        healthy_ingredients = []
        
        # Check for each known ingredient
        for ingr in TextIngredientDetector.KNOWN_INGREDIENTS:
            if ingr in text_lower:
                found_ingredients.append(ingr)
                if ingr in TextIngredientDetector.HEALTHY_STARS:
                    healthy_ingredients.append(ingr)
        
        return found_ingredients, healthy_ingredients

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
