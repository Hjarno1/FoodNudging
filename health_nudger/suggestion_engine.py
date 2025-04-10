import csv
from typing import List, Dict

from health_nudger.ingredient_detector import TextIngredientDetector

from health_nudger.nemlig_api import search_nemlig_product, get_product_details, parse_nutrition

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
                    'alternative': row['sundere_alternativ'].strip().lower(),
                    'reason': row['begrundelse'],
                    'calorie_reduction': int(row.get('kalorie_reduktion', 0)),
                    'fat_reduction': int(row.get('fedt_reduktion', 0)),
                    'sugar_reduction': int(row.get('sukker_reduktion', 0)),
                    'fiber_increase': int(row.get('fiber_forøgelse', 0)),
                    'protein_increase': int(row.get('protein_forøgelse', 0))
                }
        return subs

    def generate_nudges(self, ingredients: List[str], include_nutrition: bool = False) -> List[Dict]:
        suggestions = []
        for ingr in ingredients:
            ingr_lower = ingr.lower()
            if ingr_lower in self.substitution_dict:
                alt = self.substitution_dict[ingr_lower]['alternative']
                reason = self.substitution_dict[ingr_lower]['reason']
                # Query Nemlig.com for a product matching the alternative.
                results = search_nemlig_product(alt, take=5)
                product_info = None
                nutrition = None

                if results and results.get("Products") and results["Products"].get("Products"):
                    # Choose the first product from the results. 
                    # TODO: Handle multiple results.
                    product = results["Products"]["Products"][0]
                    product_slug = product.get("Url")  # e.g., "ciabatta-5067624"

                    # Fetch detailed product info.
                    details = get_product_details(product_slug)
                    if details:
                        if details.get("content"):
                            for block in details["content"]:
                                declaration_html = block.get("DeclarationLabel", "")
                                if "<table" in declaration_html.lower():
                                    nutrition = parse_nutrition(declaration_html)
                                    break
                        if not nutrition:
                            for block in details.get("content", []):
                                if block.get("Declarations"):
                                    dec = block.get("Declarations")
                                    nutrition = {}
                                    if dec.get("EnergyKj") or dec.get("EnergyKcal"):
                                        energy_kj = dec.get("EnergyKj", "").strip()
                                        energy_kcal = dec.get("EnergyKcal", "").strip()
                                        nutrition["Energi"] = f"{energy_kj} kJ / {energy_kcal} kcal"
                                    if dec.get("NutritionalContentFat"):
                                        nutrition["Fedt"] = f"{dec.get('NutritionalContentFat').strip()} g"
                                    if dec.get("SaturatedFattyAcid"):
                                        nutrition["heraf mættede fedtsyrer"] = f"{dec.get('SaturatedFattyAcid').strip()} g"
                                    if dec.get("NutritionalContentCarbohydrate"):
                                        nutrition["Kulhydrat"] = f"{dec.get('NutritionalContentCarbohydrate').strip()} g"
                                    if dec.get("Sugar"):
                                        nutrition["heraf sukkerarter"] = f"{dec.get('Sugar').strip()} g"
                                    if dec.get("DietaryFiber"):
                                        nutrition["Kostfibre"] = f"{dec.get('DietaryFiber').strip()} g"
                                    if dec.get("NutritionalContentProtein"):
                                        nutrition["Protein"] = f"{dec.get('NutritionalContentProtein').strip()} g"
                                    if dec.get("Salt"):
                                        nutrition["Salt"] = f"{dec.get('Salt').strip()} g"
                                    if nutrition:
                                        break
                        if not nutrition:
                            print("No nutritional data found; using fallback text.")
                    
                    product_info = {
                        "name": product.get("Name"),
                        "id": product.get("Id"),
                        "slug": product_slug,
                        "image": product.get("PrimaryImage"),
                        "nutrition": nutrition
                    }
                    
                suggestion_obj = {
                    "ingredient": ingr,
                    "alternative": alt,
                    "reason": reason,
                    "suggestionText": f"Swap '{ingr}' for '{alt}' to be healthier ({reason}).",
                    "product": product_info
                }
                suggestions.append(suggestion_obj)
            else:
                suggestions.append({
                    "ingredient": ingr,
                    "suggestionText": f"No healthier alternative found for {ingr}.",
                    "product": None
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