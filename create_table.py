from app import app, db

with app.app_context():
    db.create_all()
    print("✅ All tables created!")
