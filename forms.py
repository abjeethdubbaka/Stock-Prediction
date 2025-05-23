from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, SelectField, FloatField, IntegerField, \
    TextAreaField, DateTimeField
from wtforms.validators import DataRequired, Length, EqualTo, NumberRange, Optional, ValidationError
from models import User, TradeReason
from datetime import datetime


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=80)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=80)])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')


class TradeForm(FlaskForm):
    symbol = StringField('Symbol', validators=[DataRequired(), Length(max=10)])
    trade_type = SelectField('Trade Type', choices=[('long', 'Long'), ('short', 'Short'), ('cover', 'Cover')],
                             validators=[DataRequired()])
    entry_price = FloatField('Entry Price', validators=[DataRequired(), NumberRange(min=0)])
    exit_price = FloatField('Exit Price', validators=[Optional(), NumberRange(min=0)])
    quantity = IntegerField('Quantity', validators=[DataRequired(), NumberRange(min=1)])
    entry_date = DateTimeField('Entry Date', format='%Y-%m-%dT%H:%M:%S', default=datetime.now,
                               validators=[DataRequired()])
    exit_date = DateTimeField('Exit Date', format='%Y-%m-%dT%H:%M:%S', default=datetime.now, validators=[Optional()])
    stop_loss = FloatField('Stop Loss', validators=[Optional(), NumberRange(min=0)])
    take_profit = FloatField('Take Profit', validators=[Optional(), NumberRange(min=0)])
    fees = FloatField('Fees', validators=[Optional(), NumberRange(min=0)])
    notes = TextAreaField('Notes', validators=[Optional()])
    reason = SelectField('Reason for Trade', choices=[
        ('technical', 'Technical Analysis'),
        ('fundamental', 'Fundamental Analysis'),
        ('news', 'News Event'),
        ('other', 'Other'),
        ('emotional', 'EMOTIONAL')
    ], validators=[DataRequired()])
    submit = SubmitField('Save Trade')


class CloseTradeForm(FlaskForm):
    exit_price = FloatField('Exit Price', validators=[DataRequired(), NumberRange(min=0)])
    exit_date = DateTimeField('Exit Date', format='%Y-%m-%dT%H:%M:%S', default=datetime.now,
                              validators=[DataRequired()])
    fees = FloatField('Fees', validators=[Optional(), NumberRange(min=0)])
    notes = TextAreaField('Notes', validators=[Optional()])
    submit = SubmitField('Close Trade')


class PredictionForm(FlaskForm):
    symbol = StringField('Stock Symbol', validators=[
        DataRequired(message="Symbol is required"),
        Length(max=10)
    ], filters=[lambda x: x.strip() if x else x])
    submit = SubmitField('Predict')
