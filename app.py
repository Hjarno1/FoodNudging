import os, numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_user, logout_user,login_required, current_user
from health_nudger.bandit import LinUCB
from health_nudger.open_food_facts_api import get_product_details, search_product
from models import BanditState, UserChoice, db, User, UserPreferences
from health_nudger.ingredient_detector import TextIngredientDetector
from health_nudger.suggestion_engine import SuggestionEngine
from health_nudger.image_meal_classifier import ImageMealClassifier


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blivsund.db'
app.config['SQLALCHEMY_ECHO'] = False
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.jinja_env.globals['getattr'] = getattr


app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# bind the db instance
db.init_app(app)

# creates tables if they don't exist
with app.app_context():
    db.create_all()

login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Initialize ML

text_detector    = TextIngredientDetector()
SUB_FILE         = os.path.join("health_nudger", "data", "substitutions.csv")
suggestion_engine= SuggestionEngine(SUB_FILE)
image_classifier = ImageMealClassifier()

QUESTION_FIELDS = [
  'meat','dairy','bread','produce',
  'sustainability','swap',
  'higher_price','different_flavor','longer_prep','different_section',
  'diet','household','order_freq'
]

#register
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        pwd   = request.form['password']
        if User.query.filter_by(email=email).first():
            return "User already exists", 400

        user = User(
            email=email,
            password_hash=generate_password_hash(pwd)
        )
        db.session.add(user)
        db.session.commit()

        prefs = UserPreferences(user_id=user.id)
        db.session.add(prefs)
        db.session.commit()

        login_user(user)
        return redirect(url_for('profile'))

    return render_template('register.html')

#login
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        pwd   = request.form['password']
        user  = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, pwd):
            login_user(user)
            return redirect(url_for('index'))
        return "Invalid credentials", 401
    return render_template('login.html')

#logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for(''))

#profile
@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    prefs = current_user.profile # or .profile if you named it that
    if request.method == 'POST':
        for f in QUESTION_FIELDS:
            if f in ('higher_price','different_flavor','longer_prep','different_section'):
                setattr(prefs, f, bool(request.form.get(f)))
            else:
                setattr(prefs, f, request.form.get(f))
        db.session.commit()
        return redirect(url_for('index'))

    return render_template('profile.html', profile=prefs)


@app.route('/', methods=['GET'])
def landing():
    if current_user.is_authenticated:
        return redirect(url_for('index'))   
    return render_template('landing.html')

@app.route('/app', methods=['GET'])
@login_required
def index():
    return render_template('index.html')

@app.route('/record-choice', methods=['POST'])
@login_required
def record_choice():
    data         = request.get_json()
    ingr         = data['ingredient']          
    chosen_code  = data['product_code']         
    chosen_eco   = float(data.get('ecoscore', 0))


    alt = suggestion_engine.substitution_dict[ingr.lower()]['alternative']
    results = search_product(alt, page=1, page_size=5) or {}
    prods   = results.get("products", [])[:5]

    contexts = []
    codes    = []
    user_x   = suggestion_engine.encode_questionnaire(
                  { f: getattr(current_user.profile, f) for f in QUESTION_FIELDS }
              )
    for p in prods:
        code    = p.get("code")
        details = get_product_details(code)
        eco     = details["product"].get("ecoscore_score",0) if details and details.get("product") else 0
        x_full  = np.vstack([user_x, np.array([[eco]],dtype=float)])
        contexts.append(x_full)
        codes.append(code)

    state = BanditState.query.filter_by(
               user_id=current_user.id,
               ingredient=ingr.lower()
            ).first()
    if state:
        bandit = LinUCB.from_state(state.A_matrix, state.b_vector, alpha=0.2)
    else:
        bandit = LinUCB(n_arms=len(contexts), d=user_x.shape[0]+1, alpha=0.2)
        state  = BanditState(user_id=current_user.id, ingredient=ingr.lower())
    try:
        idx = codes.index(chosen_code)
    except ValueError:
        return jsonify({"error":"invalid product code"}), 400

    bandit.update(idx, contexts[idx], reward=1.0)

    state.A_matrix = bandit.A
    state.b_vector = bandit.b
    db.session.add(state)
    db.session.commit()

    return jsonify({"status":"ok"}), 200


@app.route('/analyze-text', methods=['POST'])
@login_required
def analyze_text():
    user_text = request.form.get('user_text','').strip()
    if not user_text:
        return jsonify({"error":"No text provided"}), 400

    prefs = current_user.profile
    questionnaire = { f: getattr(prefs, f) for f in QUESTION_FIELDS }

    ingredients  = text_detector.detect_ingredients_from_text(user_text)
    suggestions  = suggestion_engine.generate_nudges(ingredients, questionnaire)

    return jsonify({
        "detected_ingredients": ingredients,
        "suggestions": suggestions
    }), 200
    

@app.route('/analyze-image', methods=['POST'])
@login_required
def analyze_image():
    if 'meal_image' not in request.files:
        return jsonify({"error":"No image file provided"}), 400

    f = request.files['meal_image']
    if f.filename == '':
        return jsonify({"error":"Empty file name"}), 400

    filename = secure_filename(f.filename)
    path     = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)

    meal_name, ingredients = image_classifier.classify_meal_from_image(path)
    prefs = current_user.profile
    questionnaire = { f: getattr(prefs, f) for f in QUESTION_FIELDS }
    suggestions = suggestion_engine.generate_nudges(ingredients, questionnaire)

    return jsonify({
        "meal_name": meal_name,
        "detected_ingredients": ingredients,
        "suggestions": suggestions
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
