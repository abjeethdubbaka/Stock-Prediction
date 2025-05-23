from models import Trade  # Adjust import based on your structure


def calculate_metrics(user_id):
    trades = Trade.query.filter_by(user_id=user_id).all()

    if not trades:
        return {
            'win_rate': 0,
            'risk_reward': 0,
            'expectancy': 0
        }

    winning_trades = [t for t in trades if t.profit_loss and t.profit_loss > 0]
    losing_trades = [t for t in trades if t.profit_loss and t.profit_loss < 0]

    win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0

    avg_win = sum(t.profit_loss for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = abs(sum(t.profit_loss for t in losing_trades) / len(losing_trades)) if losing_trades else 0
    risk_reward = avg_win / avg_loss if avg_loss != 0 else 0

    expectancy = ((win_rate / 100) * avg_win) - ((1 - win_rate / 100) * avg_loss)

    return {
        'win_rate': round(win_rate, 1),
        'risk_reward': round(risk_reward, 2),
        'expectancy': round(expectancy, 2)
    }