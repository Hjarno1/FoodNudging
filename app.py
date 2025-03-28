import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Local imports
from health_nudger.ingredient_detector import TextIngredientDetector
from health_nudger.suggestion_engine import SuggestionEngine
from health_nudger.image_meal_classifier import ImageMealClassifier

app = Flask(__name__)

# Path to the CSV file with ingredient substitutions
SUBSTITUTION_FILE = os.path.join(os.path.dirname(__file__), 'health_nudger', 'data', 'substitutions.csv')
suggestion_engine = SuggestionEngine(SUBSTITUTION_FILE)

# Initialize detectors/classifiers
text_detector = TextIngredientDetector()
image_classifier = ImageMealClassifier()

# Simple config for uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Food Nudging API!"

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    """
    Endpoint that takes JSON with a "text" field,
    identifies ingredients, and returns nudges.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided.'}), 400

    text = data['text']
    # 1. Detect ingredients
    ingredients = text_detector.detect_ingredients_from_text(text)
    # 2. Generate nudges
    nudges = suggestion_engine.generate_nudges(ingredients)

    response = {
        'detected_ingredients': ingredients,
        'nudges': nudges
    }
    return jsonify(response), 200


@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    img_file = request.files['image']
    filename = secure_filename(img_file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    img_file.save(save_path)

    # This time we call our real classifier
    meal_name, ingredients = image_classifier.classify_meal_from_image(save_path)
    nudges = suggestion_engine.generate_nudges(ingredients)

    response = {
        'meal_name': meal_name,
        'detected_ingredients': ingredients,
        'nudges': nudges
    }
    return jsonify(response), 200


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)
