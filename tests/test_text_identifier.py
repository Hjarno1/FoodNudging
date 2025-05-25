from health_nudger.ingredient_detector import identify_ingredients_from_text

def test_text_identifier_detects_ingredient():
     found, healthy = detect_ingredients_from_text("pizza with cheese and tomato")
    assert "cheese" in result, "Text input should return 'cheese' among other ingredients."

def test_empty_text_input():
    result = identify_ingredients_from_text("")
    assert result == [], "Empty string should return empty result."
