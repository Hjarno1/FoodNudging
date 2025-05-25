# tests/test_image_processing.py

import os
from health_nudger.ingredient_detector import process_image

def test_image_returns_ingredients():
    img_path = os.path.join("uploads", "pizza.jpg")
    result = process_image(img_path)
    assert isinstance(result, list), "Result should be a list."
    assert len(result) > 0, "Image should return at least one ingredient."

def test_image_contains_known_ingredient():
    img_path = os.path.join("uploads", "pizza.jpg")
    result = process_image(img_path)
    expected = {"cheese", "tomato", "pepperoni"}
    assert any(ingredient in result for ingredient in expected), \
        f"Expected at least one of {expected} in result, got {result}"
