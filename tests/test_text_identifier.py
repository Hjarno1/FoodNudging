# tests/test_text_identifier.py

from health_nudger.ingredient_detector import detect_ingredients_from_text

def test_text_identifier_detects_ingredient():
    found, healthy = detect_ingredients_from_text("pizza with cheese and tomato")
    assert "cheese" in found, "Expected 'cheese' in detected ingredients."
    assert "tomato" in found, "Expected 'tomato' in detected ingredients."
    assert "tomato" in healthy, "Expected 'tomato' to be marked as healthy."

def test_empty_text_input():
    found, healthy = detect_ingredients_from_text("")
    assert found == [] and healthy == [], "Empty input should return two empty lists."
