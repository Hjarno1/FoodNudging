# ingest_off_data.py

import csv
from app import app, db
from models import Product
from health_nudger.open_food_facts_api import search_product

def fetch_and_store_off_products(query: str = "", pages: int = 5, page_size: int = 40, max_items: int = 10):
    with app.app_context():
        total = 0
        for page in range(1, pages + 1):
            data = search_product(query=query, page=page, page_size=page_size)
            if not data or "products" not in data:
                print(f"→ no data for page {page}")
                continue

            for prod in data["products"]:
                off_id = prod.get("code") or prod.get("id") or prod.get("_id")
                if not off_id:
                    continue
                image = (
                    prod.get("image_front_url")
                    or prod.get("image_url")
                    or prod.get("selected_images", {})
                        .get("front", {})
                        .get("display", {})
                        .get("da")
                    or prod.get("selected_images", {})
                        .get("front", {})
                        .get("sizes", {})
                        .get("400", {})
                        .get("url")
                )
                if not image:
                    continue
                ecoscore = prod.get("ecoscore_score")
                if not ecoscore:
                    continue
                p = Product.query.get(off_id) or Product(id=off_id)
                p.name            = prod.get("product_name", "")
                p.nutriments      = prod.get("nutriments", {})
                p.image_url       = prod.get("image_front_url") or prod.get("image_url")
                p.price           = None
                p.ecoscore_score  = ecoscore
                p.ecoscore_grade  = prod.get("ecoscore_grade")
                p.food_category   = query

                db.session.add(p)
                total += 1
                if total >= max_items:
                    # reached our limit—stop inserting
                    break

            db.session.commit()
            print(f"Ingested {len(data['products'])} products from page {page}")

            if total >= max_items:
                print(f"Reached target of {max_items} products. Stopping ingestion.")
                break

        print(f"Total products upserted: {total}")


def import_prices_from_csv(csv_path="prices.csv"):
    """
    (Optional) After fetching OFF data, use this to bulk-load prices.
    CSV must have headers: off_id,price
    """
    with app.app_context():
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = Product.query.get(row["off_id"])
                if not p:
                    print(f"→ skipping unknown ID {row['off_id']}")
                    continue
                try:
                    p.price = float(row["price"])
                except ValueError:
                    print(f"→ invalid price {row['price']} for {row['off_id']}")
                    continue
                db.session.add(p)
        db.session.commit()
        print(f"Prices imported from {csv_path}")


if __name__ == "__main__":
    categories = ['apples', 'sugar', 'flour', 'butter', 'beef patty', 'white bread', 'mayo', 'iceberg', 'tomato', 'cheddar', 'flour', 'tomato sauce', 'pepperoni', 'red onion', 'green pepper', 'chicken breast', 'cucumber', 'lettuce', 'mozzarella', 'pasta', 'spaghetti', 'salmon', 'tuna', 'egg', 'potato', 'carrot', 'broccoli', 'spinach', 'bell pepper']
    for category in categories:
        fetch_and_store_off_products(query=category, pages=1, page_size=40, max_items=10)

