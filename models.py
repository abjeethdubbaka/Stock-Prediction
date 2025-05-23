from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from enum import Enum as PyEnum
import pytz
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import func

db = SQLAlchemy()


class TradeType(PyEnum):
    LONG = 'long'
    SHORT = 'short'
    COVER = 'cover'


class TradeReason(PyEnum):
    TECHNICAL = 'technical'
    FUNDAMENTAL = 'fundamental'
    NEWS = 'news'
    OTHER = 'other'
    EMOTIONAL = 'emotional'


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)  # Changed from 'password'
    password = db.Column(db.Text, nullable=False)  # This seems redundant - consider removing
    trades = db.relationship('Trade', backref='user', lazy=True)
    activities = db.relationship('Activity', backref='user', lazy=True)


class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    trade_type = db.Column(db.Enum(TradeType), nullable=False)
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float)
    quantity = db.Column(db.Integer, nullable=False)
    entry_date = db.Column(db.DateTime, nullable=False)
    exit_date = db.Column(db.DateTime)
    stop_loss = db.Column(db.Float)
    take_profit = db.Column(db.Float)
    fees = db.Column(db.Float)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(pytz.UTC))
    reason = db.Column(db.Enum(TradeReason), nullable=False, default=TradeReason.OTHER)

    @hybrid_property
    def profit_loss(self):
        if self.exit_price is None:
            return None
        return (self.exit_price * self.quantity) - (self.entry_price * self.quantity) - (self.fees or 0)

    @profit_loss.expression
    def profit_loss(cls):
        return (cls.exit_price * cls.quantity) - (cls.entry_price * cls.quantity) - func.coalesce(cls.fees, 0)

    def is_win(self):
        pl = self.profit_loss  # Note: no parentheses - it's a property now
        return pl > 0 if pl is not None else False


class Activity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    action = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(pytz.UTC))
