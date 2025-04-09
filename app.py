import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

# Local imports
from health_nudger.ingredient_detector import TextIngredientDetector
from health_nudger.suggestion_engine import SuggestionEngine
from health_nudger.image_meal_classifier import ImageMealClassifier

app = Flask(__name__, )

# Setup a folder for image uploads
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



# Initialize your text ingredient detector
text_detector = TextIngredientDetector()

# Initialize your suggestion engine
SUBSTITUTION_FILE = os.path.join("health_nudger", "data", "substitutions.csv")
suggestion_engine = SuggestionEngine(SUBSTITUTION_FILE)

# Initialize your image classifier (either the mock or the real model)
image_classifier = ImageMealClassifier()

@app.route('/', methods=['GET'])
def index():
    """
    Render a simple page with two forms:
    1) For text analysis
    2) For image upload
    """
    return render_template('index.html')

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    user_text = request.form.get('user_text', '')
    if not user_text.strip():
        return jsonify({"error": "No text provided"}), 400

    # Detect ingredients (both regular and healthy)
    ingredients, healthy_ingredients = text_detector.detect_ingredients_from_text(user_text)
    
    # Generate comprehensive feedback
    feedback = suggestion_engine.generate_feedback(ingredients, healthy_ingredients)
    
    response = {
        "analysis": feedback,
        "meal_description": user_text
    }
    return jsonify(response), 200

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """
    Handle the image form submission
    """
    if 'meal_image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['meal_image']
    if file.filename == '':
        return jsonify({"error": "Empty file name"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # Classify the meal
    meal_name, ingredients = image_classifier.classify_meal_from_image(save_path)
    # Generate nudges
    nudges = suggestion_engine.generate_nudges(ingredients)

    response = {
        "meal_name": meal_name,
        "detected_ingredients": ingredients,
        "nudges": nudges
    }
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)