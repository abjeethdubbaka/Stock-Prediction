from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
from models import Trade


def generate_advanced_recommendations(user_id):
    """Generate comprehensive trading insights and recommendations"""
    # Get all closed trades
    closed_trades = Trade.query.filter(
        Trade.user_id == user_id,
        Trade.exit_date.isnot(None),
        Trade.profit_loss.isnot(None)  # Now using the hybrid property
    ).all()

    if not closed_trades:
        return {"recommendations": [], "learning_resources": []}

    # Calculate core metrics
    wins = [t for t in closed_trades if t.is_win()]
    losses = [t for t in closed_trades if not t.is_win()]
    win_rate = len(wins) / len(closed_trades) * 100

    # Time-based analysis
    recent_trades = [t for t in closed_trades if t.exit_date > datetime.utcnow() - timedelta(days=30)]
    monthly_trend = analyze_monthly_trends(closed_trades)

    # Advanced metrics
    avg_win = np.mean([t.profit_loss for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t.profit_loss) for t in losses]) if losses else 0
    risk_reward = avg_win / avg_loss if avg_loss else 0
    expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)

    # Pattern detection
    symbol_performance = analyze_symbol_performance(closed_trades)
    time_of_day_patterns = analyze_time_patterns(closed_trades)
    holding_period_analysis = analyze_holding_periods(closed_trades)


    # Generate recommendations
    recommendations = []

    # 1. Win Rate Analysis
    if win_rate < 40:
        recommendations.append({
            'type': 'risk',
            'priority': 'high',
            'title': 'Low Win Rate Detected',
            'insight': f"Your win rate is {win_rate:.1f}% (industry avg: 45-55%)",
            'recommendation': 'Improve entry timing using technical confirmation',
            'action': {
                'type': 'filter',
                'target': 'losing_trades',
                'params': {'limit': 5}
            }
        })

    # 2. Risk/Reward Analysis
    if risk_reward < 1.2:
        recommendations.append({
            'type': 'risk',
            'priority': 'high',
            'title': 'Suboptimal Risk/Reward',
            'insight': f"Your avg win ${avg_win:.2f} vs avg loss ${avg_loss:.2f} (ratio: {risk_reward:.1f}:1)",
            'recommendation': 'Aim for minimum 1.5:1 risk/reward ratio',
            'action': {
                'type': 'workshop',
                'target': 'risk_management'
            }
        })

    # 3. Time-Based Patterns
    if time_of_day_patterns.get('worst_period'):
        recommendations.append({
            'type': 'timing',
            'priority': 'medium',
            'title': 'Timing Opportunity',
            'insight': f"Trades during {time_of_day_patterns['worst_period']} underperform by {time_of_day_patterns['performance_diff']:.1f}%",
            'recommendation': 'Consider avoiding trades during this period',
            'action': {
                'type': 'visualize',
                'target': 'time_heatmap'
            }
        })
    # After calculating core metrics
    time_of_day_patterns = analyze_time_patterns(closed_trades)
    holding_period_analysis = analyze_holding_periods(closed_trades)

    # Add to recommendations
    if time_of_day_patterns.get('worst_period'):
        recommendations.append({
            'type': 'timing',
            'priority': 'medium',
            'title': 'Timing Opportunity',
            'insight': (
                f"Your {time_of_day_patterns['worst_period']} trades underperform "
                f"by {time_of_day_patterns['performance_diff']:.1f}% vs your best period"
            ),
            'recommendation': 'Consider adjusting your trading hours',
            'action': {
                'type': 'visualize',
                'target': 'time_heatmap'
            }
        })

    if holding_period_analysis.get('best_duration'):
        recommendations.append({
            'type': 'duration',
            'priority': 'medium',
            'title': 'Holding Pattern',
            'insight': (
                f"Your {holding_period_analysis['best_duration']} trades have "
                f"{holding_period_analysis['best_win_rate']:.1f}% win rate vs "
                f"{holding_period_analysis['worst_win_rate']:.1f}% for "
                f"{holding_period_analysis['worst_duration']}"
            ),
            'recommendation': 'Focus on your most successful trade durations',
            'action': {
                'type': 'filter',
                'target': 'best_duration_trades'
            }
        })
    # Generate learning resources
    learning_resources = generate_personalized_learning(
        win_rate=win_rate,
        risk_reward=risk_reward,
        expectancy=expectancy,
        patterns={
            'symbol': symbol_performance,
            'time': time_of_day_patterns,
            'holding': holding_period_analysis
        }
    )

    return {
        "recommendations": sorted(recommendations, key=lambda x: x['priority']),
        "learning_resources": learning_resources,
        "metrics": {
            "win_rate": win_rate,
            "risk_reward": risk_reward,
            "expectancy": expectancy
        }
    }


def analyze_monthly_trends(trades):
    """Analyze performance by month"""
    monthly = defaultdict(list)
    for trade in trades:
        month = trade.exit_date.strftime('%Y-%m')
        monthly[month].append(trade)

    return {
        month: {
            'win_rate': sum(1 for t in trades if t.is_win()) / len(trades),
            'profit': sum(t.profit_loss for t in trades)
        }
        for month, trades in monthly.items()
    }


def analyze_symbol_performance(trades):
    """Analyze performance by symbol"""
    symbol_data = defaultdict(list)
    for trade in trades:
        symbol_data[trade.symbol].append(trade)

    return {
        symbol: {
            'win_rate': sum(1 for t in trades if t.is_win()) / len(trades),
            'avg_profit': np.mean([t.profit_loss for t in trades]) if trades else 0
        }
        for symbol, trades in symbol_data.items()
    }


def generate_personalized_learning(**metrics):
    """Dynamically recommend learning resources"""
    resources = []

    if metrics['win_rate'] < 40:
        resources.append({
            'title': 'Mastering Trade Entries',
            'type': 'course',
            'duration': '25 min',
            'focus': 'Improving win rate',
            'url': '#'
        })

    if metrics['risk_reward'] < 1.5:
        resources.append({
            'title': 'Risk Management Deep Dive',
            'type': 'interactive',
            'duration': '15 min',
            'focus': 'Position sizing & stop losses',
            'url': 'https://www.researchgate.net/publication/377729888_Risk_Assessment_and_Management_in_Stock_Trading_Using_Artificial_Intelligence'
        })

    # Add default resources
    resources.extend([
        {
            'title': 'Journaling for Consistent Profits',
            'type': 'article',
            'duration': '8 min',
            'focus': 'Trade analysis',
            'url': 'https://vocal.media/trader/how-to-leverage-your-trading-journal-for-consistent-profits'
        },
        {
            'title': 'Market Condition Assessment',
            'type': 'cheatsheet',
            'duration': '',
            'focus': 'Adapting to volatility',
            'url': 'https://glginsights.com/articles/how-to-conduct-a-market-assessment/'
        }
    ])

    return resources


def analyze_time_patterns(trades):
    """Analyze performance by time of day"""
    time_slots = {
        'Pre-Market (4-9:30)': {'start': 4, 'end': 9.5},
        'Morning (9:30-12)': {'start': 9.5, 'end': 12},
        'Afternoon (12-16)': {'start': 12, 'end': 16},
        'Evening (16-20)': {'start': 16, 'end': 20},
        'Overnight (20-4)': {'start': 20, 'end': 4}
    }

    results = {}
    for name, slot in time_slots.items():
        slot_trades = [
            t for t in trades
            if is_time_in_slot(t.entry_date.time(), slot)
        ]

        if slot_trades:
            wins = sum(1 for t in slot_trades if t.is_win())
            win_rate = (wins / len(slot_trades)) * 100
            avg_profit = np.mean([t.profit_loss for t in slot_trades])
            results[name] = {
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'count': len(slot_trades)
            }

    # Find best/worst periods
    if results:
        best = max(results.items(), key=lambda x: x[1]['win_rate'])
        worst = min(results.items(), key=lambda x: x[1]['win_rate'])
        return {
            'best_period': best[0],
            'best_win_rate': best[1]['win_rate'],
            'worst_period': worst[0],
            'worst_win_rate': worst[1]['win_rate'],
            'performance_diff': best[1]['win_rate'] - worst[1]['win_rate'],
            'details': results
        }
    return {}


def is_time_in_slot(time_obj, slot):
    """Check if time falls within a trading slot"""
    hour = time_obj.hour + time_obj.minute / 60
    if slot['start'] < slot['end']:
        return slot['start'] <= hour < slot['end']
    else:  # Overnight slot
        return hour >= slot['start'] or hour < slot['end']


def analyze_holding_periods(trades):
    """Analyze performance by holding duration"""
    duration_buckets = {
        'Intraday (<1 day)': {'max_hours': 24},
        'Swing (1-3 days)': {'min_hours': 24, 'max_hours': 72},
        'Short-term (3-10 days)': {'min_hours': 72, 'max_hours': 240},
        'Long-term (>10 days)': {'min_hours': 240}
    }

    results = {}
    for name, bucket in duration_buckets.items():
        bucket_trades = [
            t for t in trades
            if is_in_duration_bucket(t, bucket)
        ]

        if bucket_trades:
            duration_hours = [
                (t.exit_date - t.entry_date).total_seconds() / 3600
                for t in bucket_trades
            ]
            wins = sum(1 for t in bucket_trades if t.is_win())
            results[name] = {
                'win_rate': (wins / len(bucket_trades)) * 100,
                'avg_profit': np.mean([t.profit_loss for t in bucket_trades]),
                'avg_duration': np.mean(duration_hours),
                'count': len(bucket_trades)
            }

    # Find most/least profitable durations
    if results:
        best = max(results.items(), key=lambda x: x[1]['win_rate'])
        worst = min(results.items(), key=lambda x: x[1]['win_rate'])
        return {
            'best_duration': best[0],
            'best_win_rate': best[1]['win_rate'],
            'worst_duration': worst[0],
            'worst_win_rate': worst[1]['win_rate'],
            'details': results
        }
    return {}


def is_in_duration_bucket(trade, bucket):
    """Check if trade falls within duration bucket"""
    duration_hours = (trade.exit_date - trade.entry_date).total_seconds() / 3600
    meets_min = 'min_hours' not in bucket or duration_hours >= bucket['min_hours']
    meets_max = 'max_hours' not in bucket or duration_hours < bucket['max_hours']
    return meets_min and meets_max

