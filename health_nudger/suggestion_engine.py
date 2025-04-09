import csv
from typing import List, Dict

from health_nudger.ingredient_detector import TextIngredientDetector


class SuggestionEngine:
    def __init__(self, substitution_file_path: str):
        self.substitution_dict = self._load_substitutions(substitution_file_path)

    def _load_substitutions(self, file_path: str) -> Dict[str, Dict[str, str]]:
        subs = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ingredient = row['ingredient'].strip().lower()
                subs[ingredient] = {
                    'alternative': row['healthier_alternative'].strip().lower(),
                    'reason': row['reason'],
                    'calorie_reduction': int(row.get('calorie_reduction', 0)),
                    'fat_reduction': int(row.get('fat_reduction', 0)),
                    'sugar_reduction': int(row.get('sugar_reduction', 0)),
                    'fiber_increase': int(row.get('fiber_increase', 0)),
                    'protein_increase': int(row.get('protein_increase', 0))
                }
        return subs

    def generate_nudges(self, ingredients: List[str], include_nutrition: bool = False) -> List[str]:
        suggestions = []
        for ingr in ingredients:
            ingr_lower = ingr.lower()
            if ingr_lower in self.substitution_dict:
                sub = self.substitution_dict[ingr_lower]
                alt = sub['alternative']
                reason = sub['reason']
                
                if include_nutrition:
                    nutrition_info = []
                    if sub['calorie_reduction'] > 0:
                        nutrition_info.append(f"{sub['calorie_reduction']} fewer calories")
                    if sub['fat_reduction'] > 0:
                        nutrition_info.append(f"{sub['fat_reduction']}g less fat")
                    if sub['sugar_reduction'] > 0:
                        nutrition_info.append(f"{sub['sugar_reduction']}g less sugar")
                    if sub['fiber_increase'] > 0:
                        nutrition_info.append(f"{sub['fiber_increase']}g more fiber")
                    if sub['protein_increase'] > 0:
                        nutrition_info.append(f"{sub['protein_increase']}g more protein")
                    
                    nutrition_str = " (" + ", ".join(nutrition_info) + ")" if nutrition_info else ""
                    suggestion = f"Swap '{ingr}' for '{alt}' to be healthier ({reason}){nutrition_str}."
                else:
                    suggestion = f"Swap '{ingr}' for '{alt}' to be healthier ({reason})."
                
                suggestions.append(suggestion)
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
            praise = f"Great choice with {ingr}! {TextIngredientDetector.HEALTHY_STARS.get(ingr, '')}"
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