from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from models import db, Trade, Activity, TradeType, TradeReason
from forms import TradeForm, CloseTradeForm, PredictionForm
from flask_paginate import Pagination, get_page_args
import pytz
import logging
from datetime import datetime, timedelta
from collections import namedtuple
import numpy as np
from services.analytics_service import (
    analyze_time_patterns,
    analyze_holding_periods,
    generate_advanced_recommendations,generate_personalized_learning
)
from src.prediction import process_stocks_for_predictions
from metrix import calculate_metrics
from sqlalchemy import func
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
@main_bp.route('/home')
def home():
    form = PredictionForm()
    return render_template('index.html', form=form)


@main_bp.route('/predict', methods=['POST'])
def predict():
    print("Start")
    form = PredictionForm()
    print(f"Form submitted? {form.is_submitted()}")  # Should be True
    print(f"Form validate? {form.validate()}")  # Should be True
    print(f"Form errors: {form.errors}")
    print("CSRF token exists:", 'csrf_token' in request.form)
    # print("CSRF token valid:", form._csrf.())
    if not form.is_submitted():
        flash('Invalid form submission method', 'danger')
        return redirect(url_for('main.home'))
    if form.validate_on_submit():
        print('3')
        symbol = form.symbol.data
        try:
            print('4')
            prediction = process_stocks_for_predictions(symbol)
            flash(f'Prediction for {symbol}: {prediction}', 'success')
        except Exception as e:
            flash(f'Error generating prediction: {str(e)}', 'danger')
            print("failed")
        return redirect(url_for('main.home'))
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{field}: {error}', 'danger')
        return redirect(url_for('main.home'))


@main_bp.route('/dashboard')
@login_required
def dashboard():
    # Calculate metrics (your existing code)
    metrics = calculate_metrics(current_user.id)

    # Get last 10 days of trading data
    last_10_days = []
    today = datetime.utcnow().date()

    for i in range(10):
        date = today - timedelta(days=i)

        # Query total profit/loss for each day
        daily_result = db.session.query(
            func.sum(Trade.profit_loss)
        ).filter(
            Trade.user_id == current_user.id,
            func.date(Trade.exit_time) == date
        ).scalar() or 0  # Returns 0 if no trades that day

        last_10_days.append({
            'date': date,
            'profit': float(daily_result)  # Ensure it's serializable
        })

    # Your existing queries for other data
    trades = Trade.query.filter_by(user_id=current_user.id).order_by(Trade.entry_time.desc()).limit(5).all()
    recommendations = generate_advanced_recommendations(current_user.id)
    learning_resources = generate_personalized_learning()

    return render_template(
        'overview.html',
        metrics=metrics,
        last_10_days=last_10_days,
        trades=trades,
        recommendations=recommendations,
        learning_resources=learning_resources,
        now=datetime.utcnow()
    )


@main_bp.route('/features-overview')
@login_required
def features_overview():
    # Get recent trades (last 5)
    trades = Trade.query.filter_by(user_id=current_user.id) \
        .order_by(Trade.entry_date.desc()) \
        .limit(5) \
        .all()

    # Get all trades for analytics
    all_trades = Trade.query.filter_by(user_id=current_user.id).all()

    # Calculate summary statistics
    total_trades = len(all_trades)
    closed_trades = [t for t in all_trades if t.exit_date and t.profit_loss is not None]

    # Initialize with default values if no trades
    if not closed_trades:
        time_patterns = {'details': {}}
        holding_periods = {'details': {}}
        metrics = {
            'win_rate': 0.0,
            'risk_reward': 0.0,
            'expectancy': 0.0,
            'avg_trade_size': 0.0
        }
        insights = {
            'recommendations': [],
            'learning_resources': [],
            'metrics': metrics
        }
    else:
        # Calculate core metrics
        wins = sum(1 for t in closed_trades if t.is_win())
        win_rate = (wins / len(closed_trades)) * 100

        avg_win = np.mean([t.profit_loss for t in closed_trades if t.is_win()])
        avg_loss = np.mean([abs(t.profit_loss) for t in closed_trades if not t.is_win()])
        risk_reward = avg_win / avg_loss if avg_loss else 0
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)

        # Calculate average trade size (absolute value)
        avg_trade_size = np.mean([abs(t.profit_loss) for t in closed_trades])

        metrics = {
            'win_rate': win_rate,
            'risk_reward': risk_reward,
            'expectancy': expectancy,
            'avg_trade_size': avg_trade_size
        }

        # Generate advanced analytics
        time_patterns = analyze_time_patterns(closed_trades)
        holding_periods = analyze_holding_periods(closed_trades)
        insights = generate_advanced_recommendations(current_user.id)

    # Create summary namedtuple
    Summary = namedtuple('Summary', [
        'total_trades',
        'wins',
        'losses',
        'win_rate',
        'total_profit'
    ])

    wins_count = sum(1 for t in closed_trades if t.is_win())
    total_profit = sum(t.profit_loss for t in closed_trades if t.profit_loss is not None)

    summary = Summary(
        total_trades=total_trades,
        wins=wins_count,
        losses=len(closed_trades) - wins_count,
        win_rate=metrics.get('win_rate', 0),
        total_profit=total_profit
    )

    # Prepare performance data for the chart
    performance_data = {'dates': [], 'cumulative_profit': []}
    if closed_trades:
        closed_trades_sorted = sorted(closed_trades, key=lambda x: x.exit_date)
        running_total = 0
        for trade in closed_trades_sorted:
            performance_data['dates'].append(trade.exit_date.strftime('%Y-%m-%d'))
            running_total += trade.profit_loss
            performance_data['cumulative_profit'].append(round(running_total, 2))

    return render_template(
        'features/overview.html',
        trades=trades,
        summary=summary,
        performance_data=performance_data,
        recommendations=insights.get('recommendations', []),
        learning_resources=insights.get('learning_resources', []),
        metrics=metrics,
        time_patterns=time_patterns,
        holding_periods=holding_periods,
        now=datetime.utcnow()
    )


@main_bp.route('/features/charts')
@login_required
def charts():
    trades = Trade.query.filter_by(user_id=current_user.id).all()
    return render_template('features/charts.html', trades=trades)


@main_bp.route('/trading-records')
@login_required
def trading_records():
    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    per_page = 10
    total = Trade.query.filter_by(user_id=current_user.id).count()
    trades = Trade.query.filter_by(user_id=current_user.id).order_by(Trade.created_at.desc()).offset(offset).limit(
        per_page).all()
    pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap5')
    return render_template('trades/trading_records.html', trades=trades, pagination=pagination)


@main_bp.route('/trade/new', methods=['GET', 'POST'])
@login_required
def new_trade():
    form = TradeForm()
    if form.validate_on_submit():
        logger.debug(f"Entry Date: {form.entry_date.data}")
        logger.debug(f"Exit Date: {form.exit_date.data}")
        logger.debug(f"Reason: {form.reason.data}")
        trade = Trade(
            user_id=current_user.id,
            symbol=form.symbol.data,
            trade_type=TradeType(form.trade_type.data),
            entry_price=form.entry_price.data,
            exit_price=form.exit_price.data,
            quantity=form.quantity.data,
            entry_date=form.entry_date.data or datetime.now(pytz.UTC),
            exit_date=form.exit_date.data or None,
            stop_loss=form.stop_loss.data,
            take_profit=form.take_profit.data,
            fees=form.fees.data,
            notes=form.notes.data,
            reason=TradeReason(form.reason.data)
        )
        db.session.add(trade)
        db.session.commit()
        flash('Trade added successfully!', 'success')
        return redirect(url_for('main.trading_records'))
    else:
        for field, errors in form.errors.items():
            for error in errors:
                logger.error(f"Form error in {field}: {error}")
    return render_template('trades/trade_form.html', form=form, title='New Trade')


@main_bp.route('/trade/<int:trade_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_trade(trade_id):
    trade = Trade.query.get_or_404(trade_id)
    if trade.user_id != current_user.id:
        flash('Unauthorized action.', 'danger')
        return redirect(url_for('main.trading_records'))
    form = TradeForm(obj=trade)
    if form.validate_on_submit():
        logger.debug(f"Edit - Entry Date: {form.entry_date.data}")
        logger.debug(f"Edit - Exit Date: {form.exit_date.data}")
        logger.debug(f"Edit - Reason: {form.reason.data}")
        trade.symbol = form.symbol.data
        trade.trade_type = TradeType(form.trade_type.data)
        trade.entry_price = form.entry_price.data
        trade.exit_price = form.exit_price.data
        trade.quantity = form.quantity.data
        trade.entry_date = form.entry_date.data or datetime.now(pytz.UTC)
        trade.exit_date = form.exit_date.data or None
        trade.stop_loss = form.stop_loss.data
        trade.take_profit = form.take_profit.data
        trade.fees = form.fees.data
        trade.notes = form.notes.data
        trade.reason = TradeReason(form.reason.data)
        db.session.commit()
        flash('Trade updated successfully!', 'success')
        return redirect(url_for('main.trading_records'))
    else:
        for field, errors in form.errors.items():
            for error in errors:
                logger.error(f"Form error in {field}: {error}")
    return render_template('trades/edit_trade.html', form=form, trade=trade, title='Edit Trade')


@main_bp.route('/trade/<int:trade_id>/close', methods=['GET', 'POST'])
@login_required
def close_trade(trade_id):
    trade = Trade.query.get_or_404(trade_id)
    if trade.user_id != current_user.id:
        flash('Unauthorized action.', 'danger')
        return redirect(url_for('main.trading_records'))
    form = CloseTradeForm()
    if form.validate_on_submit():
        logger.debug(f"Exit Date: {form.exit_date.data}")
        trade.exit_price = form.exit_price.data
        trade.exit_date = form.exit_date.data or datetime.now(pytz.UTC)
        trade.fees = form.fees.data or trade.fees
        trade.notes = form.notes.data or trade.notes
        db.session.commit()
        flash('Trade closed successfully!', 'success')
        return redirect(url_for('main.trading_records'))
    else:
        for field, errors in form.errors.items():
            for error in errors:
                logger.error(f"Form error in {field}: {error}")
    return render_template('trades/close_trade.html', form=form, trade=trade)


@main_bp.route('/analytics')
@login_required
def analytics():
    trades = Trade.query.filter_by(user_id=current_user.id).all()
    return render_template('analytics.html', trades=trades)


