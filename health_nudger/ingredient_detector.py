import re
from typing import List, Tuple

class TextIngredientDetector:

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

    @staticmethod
    def detect_ingredients_from_text(text: str) -> Tuple[List[str], List[str]]:
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

    def classify_meal_from_image(self, image_path: str) -> Tuple[str, List[str]]:
        if "pizza" in image_path.lower():
            return ("Pepperoni Pizza", ["pepperoni", "full-fat cheese"])
        elif "burger" in image_path.lower():
            return ("Cheeseburger", ["full-fat cheese", "mayo", "white bread"])
        else:
            # default mock
            return ("Salad", ["lettuce", "tomato", "chicken"])
