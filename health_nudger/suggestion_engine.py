import csv
from typing import List, Dict

class SuggestionEngine:
    """
    Loads a small knowledge base of (ingredient -> healthier alternative).
    Generates suggestions for each recognized ingredient, if applicable.
    """

    def __init__(self, substitution_file_path: str):
        self.substitution_dict = self._load_substitutions(substitution_file_path)

    def _load_substitutions(self, file_path: str) -> Dict[str, Dict[str, str]]:
        """
        Reads a CSV with columns: ingredient, healthier_alternative, reason
        Returns a dictionary:
            {
                'mayo': {
                    'healthier_alternative': 'greek yogurt',
                    'reason': 'lower fat and added protein'
                },
                'white bread': {
                    ...
                }
                ...
            }
        """
        subs = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ingredient = row['ingredient'].strip().lower()
                subs[ingredient] = {
                    'alternative': row['healthier_alternative'].strip().lower(),
                    'reason': row['reason']
                }
        return subs

    def generate_nudges(self, ingredients: List[str]) -> List[str]:
        """
        For each recognized ingredient, if there's a healthier alternative in the dictionary,
        generate a suggestion string.
        """
        suggestions = []
        for ingr in ingredients:
            ingr_lower = ingr.lower()
            if ingr_lower in self.substitution_dict:
                alt = self.substitution_dict[ingr_lower]['alternative']
                reason = self.substitution_dict[ingr_lower]['reason']
                suggestion = (f"Swap '{ingr}' for '{alt}' to be healthier ({reason}).")
                suggestions.append(suggestion)
        return suggestions
