import requests

def search_product(query, page=1, page_size=40):
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        "search_terms": query,
        "search_simple": 1,
        "action": "process",
        "page": page,
        "page_size": page_size,
        "json": 1,
        "tagtype_0": "countries",
        "tag_contains_0": "contains",
        "tag_0": "Denmark"
    }
    response = requests.get(url, params=params)
    if response.ok:
        return response.json()
    return None

def get_product_details(code):
    url = f"https://world.openfoodfacts.org/api/v0/product/{code}.json"
    response = requests.get(url)
    if response.ok:
        return response.json()
    return None

def parse_nutrition(off_product):
    product = off_product.get("product", {})
    nutriments = product.get("nutriments", {})
    nutrition_data = {}

    if "energy_100g" in nutriments and "energy-kcal_100g" in nutriments:
        nutrition_data["Energi"] = f"{nutriments['energy_100g']} kJ / {nutriments['energy-kcal_100g']} kcal"
    if "fat_100g" in nutriments:
        nutrition_data["Fedt"] = f"{nutriments['fat_100g']} g"
    if "saturated-fat_100g" in nutriments:
        nutrition_data["Heraf mættede fedtsyrer"] = f"{nutriments['saturated-fat_100g']} g"
    if "carbohydrates_100g" in nutriments:
        nutrition_data["Kulhydrat"] = f"{nutriments['carbohydrates_100g']} g"
    if "sugars_100g" in nutriments:
        nutrition_data["Heraf sukkerarter"] = f"{nutriments['sugars_100g']} g"
    if "fiber_100g" in nutriments:
        nutrition_data["Kostfibre"] = f"{nutriments['fiber_100g']} g"
    if "proteins_100g" in nutriments:
        nutrition_data["Protein"] = f"{nutriments['proteins_100g']} g"
    if "salt_100g" in nutriments:
        nutrition_data["Salt"] = f"{nutriments['salt_100g']} g"
    if "ecoscore_grade" in product:
        nutrition_data["Ecoscore Grade"] = product["ecoscore_grade"]
    if "ecoscore_score" in product:
        nutrition_data["Ecoscore Score"] = product["ecoscore_score"]

    return nutrition_data
