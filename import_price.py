# import_prices.py
import csv
from app import app, db
from models import Product

def import_prices_from_csv(csv_path="prices.csv"):
    """
    CSV must have headers: off_id,price
    """
    with app.app_context():
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = Product.query.get(row['off_id'])
                if not p:
                    print(f"→ skipping unknown ID {row['off_id']}")
                    continue
                try:
                    p.price = float(row['price'])
                except ValueError:
                    print(f"→ invalid price {row['price']} for {row['off_id']}")
                    continue
                db.session.add(p)
        db.session.commit()
        print(f"Imported prices from {csv_path}")

if __name__ == "__main__":
    import_prices_from_csv()
