import csv
from flask_login import current_user
import numpy as np
from typing import List, Dict

from models import Product, BanditState, db
from health_nudger.bandit import LinUCB

class SuggestionEngine:
    """
    Loads substitution mappings from CSV and generates a single, personalized
    healthier-alternative suggestion per detected ingredient using a LinUCB bandit.
    """

    def __init__(self, substitution_file_path: str, alpha: float = 0.2):
        self.alpha = alpha
        self.substitution_dict = self._load_substitutions(substitution_file_path)

    def _load_substitutions(self, file_path: str) -> Dict[str, Dict[str, str]]:
        subs = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ingr = row['ingredient'].strip().lower()
                subs[ingr] = {
                    'alternative': row['healthier_alternative'].strip().lower(),
                    'reason': row['reason']
                }
        return subs

    def encode_questionnaire(self, q: Dict) -> np.ndarray:
        freq_map = {'0-1': 0, '2-3': 1, '4-5': 2, '6+': 3}
        diet_options = ['none','vegetarian','vegan','pescatarian','flexitarian','other']
        feats = []
        for key in ['Meat','Dairy','Bread','Produce']:
            feats.append(freq_map.get(q.get(key, '0-1'), 0))
        feats.append(int(q.get('Sustainability', 3)))
        for opt in ['yes_similar','yes_cost','maybe_time','no']:
            feats.append(1 if q.get('Swap') == opt else 0)
        for t in ['higher_price','different_flavor','longer_prep','different_section']:
            feats.append(1 if q.get(t) else 0)
        for dopt in diet_options:
            feats.append(1 if q.get('Diet') == dopt else 0)
        house_map = {'1':1,'2':2,'3':3,'4+':4}
        order_map = {'Weekly':4,'Bi-weekly':3,'Monthly':2,'Rarely/Never':1}
        feats.append(house_map.get(q.get('Household'), 1))
        feats.append(order_map.get(q.get('OrderFreq'), 1))
        return np.array(feats, dtype=float).reshape(-1,1)

    def _get_products(self, alternative: str, limit: int = 10) -> List[Product]:
        return (
            Product.query
                   .filter(Product.food_category.ilike(f"%{alternative}%"))
                   .limit(limit)
                   .all()
        )

    def generate_nudges(self, ingredients: List[str], questionnaire: Dict) -> List[Dict]:
        suggestions: List[Dict] = []
        user_x = self.encode_questionnaire(questionnaire)
        user_dim = user_x.shape[0]

        for ingr in ingredients:
            ingr_key = ingr.lower()
            if ingr_key not in self.substitution_dict:
                suggestions.append({
                    "ingredient":      ingr,
                    "alternative":     None,
                    "reason":          None,
                    "chosen_products": []
                })
                continue

            alt    = self.substitution_dict[ingr_key]['alternative']
            reason = self.substitution_dict[ingr_key]['reason']
            prods  = self._get_products(alt, limit=10)

            contexts = []
            arms      = []
            for p in prods:
                eco   = float(p.ecoscore_score or 0.0)
                price = float(p.price        or 0.0)
                # stack [user x | ecoscore | price]
                x_full = np.vstack([
                    user_x,
                    np.array([[eco]],   dtype=float),
                    np.array([[price]], dtype=float),
                ])
                contexts.append(x_full)
                arms.append({
                    "code":      p.id,
                    "name":      p.name,
                    "image":     p.image_url,
                    "nutrition": p.nutriments,
                    "ecoscore":  eco,
                    "price":     price
                })

            if not contexts:
                suggestions.append({
                    "ingredient":      ingr,
                    "alternative":     alt,
                    "reason":          reason,
                    "chosen_products": []
                })
                continue

            # load or init bandit state
            wanted_dim  = user_dim + 2      # user_x + ecoscore + price
            n_arms_now  = len(contexts)

            state = BanditState.query.filter_by(ingredient=ingr_key, user_id = current_user.id,).first()
            if state:
                A_list = state.A_matrix              # this is a list of np.ndarray
                # grab any one matrix to check its dimension
                existing_dim = A_list[0].shape[0]     # eg (22,22) before, now should be (23,23)
                existing_arms = len(A_list)

                # if either the feature-dim or the number of arms has changed, discard old state
                if existing_dim != wanted_dim or existing_arms != n_arms_now:
                    db.session.delete(state)
                    db.session.commit()
                    state = None

            if state:
                bandit = LinUCB.from_state(state.A_matrix, state.b_vector, alpha=self.alpha)
            else:
                bandit = LinUCB(n_arms=n_arms_now, d=wanted_dim, alpha=self.alpha)
                state  = BanditState(
                    ingredient=ingr_key,
                    user_id=current_user.id,
                    A_matrix=bandit.A,
                    b_vector=bandit.b
                )
                db.session.add(state)

            # score & pick top-3 arms
            p_vals   = bandit.ucb_scores(contexts)
            top_idxs = sorted(range(len(p_vals)),
                              key=lambda i: p_vals[i],
                              reverse=True)[:3]
            chosen = [arms[i] for i in top_idxs]

            suggestions.append({
                "ingredient":      ingr,
                "alternative":     alt,
                "reason":          reason,
                "chosen_products": chosen
            })

        db.session.commit()
        return suggestions

    