import os, numpy as np
from typing import List
from flask import Flask, render_template, request, jsonify, redirect, url_for, current_app as app
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_user, logout_user,login_required, current_user
from health_nudger.bandit import LinUCB
from health_nudger.open_food_facts_api import get_product_details, search_product
from models import BanditState, Product, UserChoice, db, User, UserPreferences
from health_nudger.ingredient_detector import TextIngredientDetector
from health_nudger.suggestion_engine import SuggestionEngine
from health_nudger.image_meal_classifier import ImageMealClassifier


app = Flask(__name__)
app.jinja_env.globals['getattr'] = getattr
app.config.update({
    'SECRET_KEY':                'secret_key',
    'SQLALCHEMY_DATABASE_URI':   'sqlite:///blivsund.db',
    'SQLALCHEMY_ECHO':           False,
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'UPLOAD_FOLDER':             'uploads'
})
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db.init_app(app)
with app.app_context():
    db.create_all()

login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

text_detector     = TextIngredientDetector()
SUB_FILE          = os.path.join("health_nudger","data","substitutions.csv")
suggestion_engine = SuggestionEngine(SUB_FILE)

FIELD_MAP = {
  'Meat': 'meat',
  'Dairy': 'dairy',
  'Bread': 'bread',
  'Produce': 'produce',
  'Sustainability': 'sustainability',
  'Swap': 'swap',
  'higher_price': 'higher_price',
  'different_flavor': 'different_flavor',
  'longer_prep': 'longer_prep',
  'different_section': 'different_section',
  'Diet': 'diet',
  'Household': 'household',
  'OrderFreq': 'order_freq',
}

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method=='POST':
        email, pwd = request.form['email'], request.form['password']
        if User.query.filter_by(email=email).first():
            return "User exists",400
        user = User(
            email=email,
            password_hash=generate_password_hash(pwd)
        )
        db.session.add(user); db.session.commit()
        prefs = UserPreferences(user_id=user.id)
        db.session.add(prefs); db.session.commit()
        login_user(user)
        return redirect(url_for('profile'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        u = User.query.filter_by(email=request.form['email']).first()
        if u and check_password_hash(u.password_hash,request.form['password']):
            login_user(u)
            return redirect(url_for('index'))
        return "Invalid credentials",401
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    prefs = current_user.profile
    if request.method=='POST':
        for f in FIELD_MAP:
            val = request.form.get(f)
            if f in ('higher_price','different_flavor','longer_prep','different_section'):
                setattr(prefs, f, bool(val))
            else:
                setattr(prefs, f, val)
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('profile.html', profile=prefs)

@app.route('/get-products', methods=['POST'])
def get_products_view():
    # 1) Try to parse JSON, fall back to empty dict
    data = request.get_json(force=True) or {}
    app.logger.debug("get-products payload: %r", data)

    # 2) Make sure we have the override text
    alternative = data.get('alternative')
    if not alternative:
        app.logger.error("no alternative in payload")
        return jsonify(error="No alternative provided"), 400

    try:
        # 3) Fetch from your suggestion engine
        prods = suggestion_engine._get_products(alternative, limit=3)
        app.logger.debug("found products: %r", [p.id for p in prods])

        # 4) Serialize
        chosen = []
        for p in prods:
            app.logger.debug("serializing product %s", p.id)
            chosen.append({
                "code":     p.id,
                "name":     p.name,
                "image":    getattr(p, "image_url", None),
                "ecoscore": p.ecoscore_score,
                "price":    p.price
            })

        # 5) Return JSON only
        return jsonify(chosen_products=chosen), 200

    except Exception:
        # Log full traceback
        app.logger.exception("Error in get-products")
        return jsonify(error="Internal server error"), 500



@app.route('/')
def landing():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('landing.html')

@app.route('/app')
@login_required
def index():
    return render_template('index.html')

@app.route('/record-choice', methods=['POST'])
@login_required
def record_choice():
    data        = request.get_json()
    ingr     = data.get('ingredient', '').lower().strip()
    chosen_code = data['product_code'].strip()

    # build contexts from DB
    questionnaire = {
        frontend_key: getattr(current_user.profile, model_attr)
        for frontend_key, model_attr in FIELD_MAP.items()
    }
    user_x   = suggestion_engine.encode_questionnaire(questionnaire)
    user_dim = user_x.shape[0]

    # grab the 5 candidates
    alt   = suggestion_engine.substitution_dict[ingr]['alternative']
    prods = suggestion_engine._get_products(alt)

    contexts, codes = [], []
    for p in prods:
        eco   = float(p.ecoscore_score or 0.0)
        price = float(p.price or 0.0)
        x_full = np.vstack([
            user_x,
            [[eco]],
            [[price]],
        ])
        contexts.append(x_full)
        codes.append(p.id)

    # load or init bandit
    state = BanditState.query.filter_by(user_id = current_user.id,ingredient=ingr).first()
    if state:
        bandit = LinUCB.from_state(state.A_matrix, state.b_vector, alpha=0.2)
    else:
        bandit = LinUCB(n_arms=len(contexts), d=user_dim+2, alpha=0.2)
        state = BanditState(ingredient=ingr,
                            user_id=current_user.id,
                            A_matrix=bandit.A,
                            b_vector=bandit.b)

    # update on the chosen arm
    try:
        idx = codes.index(chosen_code)
    except ValueError:
        # user‐typed override: just record it and skip bandit update
        from models import UserChoice
        choice = UserChoice(
            user_id=current_user.id,
            ingredient=ingr,
            product_code=chosen_code
        )
        db.session.add(choice)
        db.session.commit()
        return jsonify(status="ok—override"),200

    bandit.update(idx, contexts[idx], reward=1.0)
    state.A_matrix = bandit.A
    state.b_vector = bandit.b
    db.session.add(state)
    db.session.commit()

    return jsonify(status="ok"),200

@app.route('/analyze-text', methods=['POST'])
@login_required
def analyze_text():
    user_text   = request.form.get('user_text','').strip()
    ingredients = text_detector.detect_ingredients_from_text(user_text)
    questionnaire = {
    front_key: getattr(current_user.profile, model_attr)
    for front_key, model_attr in FIELD_MAP.items()
    }
    suggestions = suggestion_engine.generate_nudges(ingredients, questionnaire)
    return jsonify(
        detected_ingredients=ingredients,
        suggestions=suggestions
    ),200

@app.route('/analyze-image', methods=['POST'])
@login_required
def analyze_image():
    if 'meal_image' not in request.files:
        return jsonify(error="No image"),400
    f = request.files['meal_image']
    if f.filename=='':
        return jsonify(error="Empty file name"),400
    fn   = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
    f.save(path)

    prefs = current_user.profile

    # classify meal & ingredients
    from health_nudger.image_meal_classifier import ImageMealClassifier
    image_classifier = ImageMealClassifier()
    meal_name, ingredients = image_classifier.classify_meal_from_image(path)

    questionnaire = {
        front_key: getattr(current_user.profile, model_attr)
        for front_key, model_attr in FIELD_MAP.items()
    }
    suggestions = suggestion_engine.generate_nudges(ingredients, questionnaire)

    return jsonify(
        meal_name=meal_name,
        detected_ingredients=ingredients,
        suggestions=suggestions
    ),200

@app.route('/account')
@login_required
def account():
    return render_template('account.html')

if __name__=='__main__':
    app.run(debug=True, port=5000)