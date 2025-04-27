import requests
from bs4 import BeautifulSoup

API_ROOT = 'https://www.nemlig.com/webapi'

def search_nemlig_product(product_query, take=5):
    url = f"{API_ROOT}/s/0/1/0/Search/Search"
    params = {'query': product_query, 'take': take}
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, params=params, headers=headers)
    if response.ok:
        return response.json()
    return None

def get_product_details(product_slug):
    url = f"https://www.nemlig.com/{product_slug}?GetAsJson=1"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    if response.ok:
        details = response.json()
        # Debug: print all content blocks and their keys
        details = response.json()
        for block in details.get("content", []):
            if block.get("TemplateName") == "productdetailspot":
                declaration_html = block.get("DeclarationLabel")
                print("Full DeclarationLabel HTML length:", len(declaration_html))
                print("Full DeclarationLabel HTML:", declaration_html)

        return details
    return None


def parse_nutrition(declaration_label_html):
    print("Parsed nutrition:", declaration_label_html)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(declaration_label_html, 'html.parser')
    nutrition_data = {}

    rows = soup.select("table tr")
    for row in rows:
        if row.find("th"):
            continue
        cells = row.find_all("td")
        if len(cells) >= 2:
            nutrient = cells[0].get_text(strip=True)
            value = cells[1].get_text(strip=True)
            if len(cells) > 2:
                unit = cells[2].get_text(strip=True)
                nutrition_data[nutrient] = f"{value} {unit}".strip()
            else:
                nutrition_data[nutrient] = value
                
                

    if not nutrition_data:
        print("Parsed nutrition:", declaration_label_html)
        fallback_text = soup.get_text(" ", strip=True)
        return {"details": fallback_text}
    
    return nutrition_data




