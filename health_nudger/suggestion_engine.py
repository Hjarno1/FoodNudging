import csv
from typing import List, Dict

from flask_login import current_user
from health_nudger.open_food_facts_api import search_product, get_product_details, parse_nutrition
from health_nudger.ingredient_detector import TextIngredientDetector
from health_nudger.bandit import LinUCB
import numpy as np

from models import BanditState

class SuggestionEngine:
    """
    Loads substitution mappings from CSV and generates a single, personalized
    healthier‑alternative suggestion per detected ingredient using a LinUCB bandit.
    """
    def __init__(self, substitution_file_path: str):
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
        #purchase frequencies
        for key in ['Meat','Dairy','Bread','Produce']:
            feats.append(freq_map.get(q.get(key, '0-1'), 0))
        #sustainability importance
        feats.append(int(q.get('Sustainability', 3)))
        #willingness to nudge
        for opt in ['yes_similar','yes_cost','maybe_time','no']:
            feats.append(1 if q.get('Swap') == opt else 0)
        #acceptable trade‑offs
        for t in ['higher_price','different_flavor','longer_prep','different_section']:
            feats.append(1 if q.get(t) else 0)
        #diets
        for dopt in diet_options:
            feats.append(1 if q.get('Diet') == dopt else 0)
        #household size & order frequency
        house_map = {'1':1,'2':2,'3':3,'4+':4}
        order_map = {'Weekly':4,'Bi‑weekly':3,'Monthly':2,'Rarely/Never':1}
        feats.append(house_map.get(q.get('Household'), 1))
        feats.append(order_map.get(q.get('OrderFreq'), 1))
        return np.array(feats, dtype=float).reshape(-1,1)

    def generate_nudges(self, ingredients: List[str], questionnaire: Dict) -> List[Dict]:
        suggestions = []
        user_x   = self.encode_questionnaire(questionnaire)
        user_dim = user_x.shape[0]

        for ingr in ingredients:
            ingr_key = ingr.lower()
            if ingr_key not in self.substitution_dict:
                suggestions.append({"ingredient": ingr, "chosen_product": None})
                continue

            alt    = self.substitution_dict[ingr_key]['alternative']
            reason = self.substitution_dict[ingr_key]['reason']
            results = search_product(alt, page=1, page_size=5) or {}
            prods   = results.get("products", [])[:5]

            contexts = []
            arms     = []
            for p in prods:
                code    = p.get("code")
                details = get_product_details(code)
                eco     = details["product"].get("ecoscore_score",0) if details and details.get("product") else 0
                x_full  = np.vstack([user_x, np.array([[eco]],dtype=float)])
                contexts.append(x_full)
                arms.append({
                    "code": code,
                    "name": p.get("product_name"),
                    "image": p.get("image_front_url"),
                    "nutrition": parse_nutrition(details) if details else {},
                    "ecoscore": eco
                })

            if not contexts:
                suggestions.append({
                "ingredient": ingr,
                "alternative": alt,
                "reason": reason,
                "chosen_product": None
                })
                continue

            #load and init bandit for (user,ingredient)
            state = BanditState.query.filter_by(
                    user_id=current_user.id,
                    ingredient=ingr_key
                ).first()
            if state:
                bandit = LinUCB.from_state(state.A_matrix, state.b_vector, alpha=0.2)
            else:
                bandit = LinUCB(n_arms=len(contexts), d=user_dim+1, alpha=0.2)

            # pick the 3 arms by UCB 
            p_vals   = bandit.ucb_scores(contexts)
            top_idxs = sorted(range(len(p_vals)),
                            key=lambda i: p_vals[i],
                            reverse=True)[:3]

            chosen_products = [ arms[i] for i in top_idxs ]

            suggestions.append({
                "ingredient":      ingr,
                "alternative":     alt,
                "reason":          reason,
                "chosen_products": chosen_products
            })

        return suggestions

    
    def get_detailed_substitutions(self, ingredients: List[str]) -> List[Dict]:
        detailed_subs = []
        for ingr in ingredients:
            ingr_lower = ingr.lower()
            if ingr_lower in self.substitution_dict:
                sub_data = self.substitution_dict[ingr_lower].copy()
                sub_data['original'] = ingr
                detailed_subs.append(sub_data)
        return detailed_subs

    def generate_feedback(self, ingredients: List[str], healthy_ingredients: List[str]) -> dict:
        nudges = self.generate_nudges(ingredients, include_nutrition=True)
        praises = []
        for ingr in healthy_ingredients:
            praise = f"Great choice with {ingr}!"
            praises.append(praise)
        assessment = self._generate_assessment(len(nudges), len(ingredients), len(praises))
        
        return {
            "suggestions": nudges,
            "praise": praises,
            "assessment": assessment,
            "healthy_ingredients": healthy_ingredients,
            "all_ingredients": ingredients
        }
    
    def _generate_assessment(self, num_suggestions: int, total_ingredients: int, num_praises: int) -> str:
        if num_suggestions == 0 and num_praises > 0:
            return "Excellent meal choice! You're already making healthy selections."
        elif num_suggestions == 0:
            return "This seems like a balanced meal. No substitutions needed!"
        elif num_suggestions / total_ingredients > 0.5:
            return "This meal could use some healthier alternatives. Consider the suggestions below."
        else:
            return "Good foundation with room for some healthy improvements!"