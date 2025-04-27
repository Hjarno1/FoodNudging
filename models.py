import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    bandit_states = db.relationship('BanditState', back_populates='user', lazy='dynamic')
    profile = db.relationship('UserPreferences', uselist=False, back_populates='user')
    choices = db.relationship('UserChoice', back_populates='user', lazy='dynamic')

class UserPreferences(db.Model):
    __tablename__ = 'user_preferences'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True)
    meat = db.Column(db.String(4), default='0-1')
    dairy = db.Column(db.String(4), default='0-1')
    bread = db.Column(db.String(4), default='0-1')
    produce = db.Column(db.String(4), default='0-1')
    sustainability = db.Column(db.Integer, default=3)
    swap = db.Column(db.String(16), default='yes_similar')
    higher_price = db.Column(db.Boolean, default=False)
    different_flavor = db.Column(db.Boolean, default=False)
    longer_prep = db.Column(db.Boolean, default=False)
    different_section = db.Column(db.Boolean, default=False)
    diet = db.Column(db.String(16), default='none')
    household = db.Column(db.String(4), default='1')
    order_freq = db.Column(db.String(16), default='Weekly')
    user = db.relationship('User', back_populates='profile')

class UserChoice(db.Model):
    __tablename__ = 'user_choices'
    id  = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    ingredient = db.Column(db.String, nullable=False)
    product_code = db.Column(db.String, nullable=False)
    ecoscore = db.Column(db.String, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.UTC)
    user = db.relationship('User', back_populates='choices')



class BanditState(db.Model):
    __tablename__ = 'bandit_states'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    ingredient  = db.Column(db.String, nullable=False)
    A_matrix    = db.Column(db.PickleType, nullable=False)
    b_vector    = db.Column(db.PickleType, nullable=False)
    user = db.relationship('User', back_populates='bandit_states')
