import json
import numpy as np

entries = []
with open('debug-627897.log', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except:
                pass

ep_ends = [e for e in entries if e.get('location') == 'trading_env.py:EPISODE_END']
print(f'Total EPISODE_END entries: {len(ep_ends)}')

# Last 3 are validation slices
val_eps = ep_ends[-3:]

print('\n' + '='*65)
print('  TRAINING LOG DIAGNOSTIC')
print('='*65)

for i, ep in enumerate(val_eps):
    d  = ep['data']
    ss = d.get('sell_summary', {})
    ec = d.get('trade_economics', {})

    print(f'\n--- Validation Slice {i} ---')
    print(f'  Trades      : {d.get("total_trades")}  |  Steps: {d.get("episode_steps")}')
    print(f'  Winners     : {ss.get("n_winners")}  |  Losers: {ss.get("n_losers")}')
    print()

    # PnL quality
    mwp  = (ss.get("mean_winner_pnl_pct")   or 0) * 100
    mlp  = (ss.get("mean_loser_pnl_pct")    or 0) * 100
    medwp = (ss.get("median_winner_pnl_pct") or 0) * 100
    medlp = (ss.get("median_loser_pnl_pct")  or 0) * 100
    bigw  = (ss.get("largest_winner_pnl_pct") or 0) * 100
    bigl  = (ss.get("largest_loser_pnl_pct")  or 0) * 100
    bigwg = ss.get("largest_winner_gross") or 0
    biglg = ss.get("largest_loser_gross")  or 0

    print(f'  PnL QUALITY (Case 1 vs 2):')
    print(f'    Mean  winner: {mwp:+.3f}%   |  Mean  loser: {mlp:+.3f}%')
    print(f'    Median winner:{medwp:+.3f}%   |  Median loser:{medlp:+.3f}%')
    print(f'    Biggest winner: {bigw:+.3f}% (${bigwg:.2f})')
    print(f'    Biggest loser:  {bigl:+.3f}% (${biglg:.2f})')
    print()

    # Hold duration by outcome
    mhw  = ss.get("mean_hold_winner")   or 0
    mhl  = ss.get("mean_hold_loser")    or 0
    medhw = ss.get("median_hold_winner") or 0
    medhl = ss.get("median_hold_loser")  or 0
    maxhw = ss.get("max_hold_winner")    or 0
    maxhl = ss.get("max_hold_loser")     or 0

    print(f'  HOLD DURATION by outcome:')
    print(f'    Mean   winner: {mhw:.0f}c  |  Mean   loser: {mhl:.0f}c')
    print(f'    Median winner: {medhw:.0f}c  |  Median loser: {medhl:.0f}c')
    print(f'    Max    winner: {maxhw}c  |  Max    loser: {maxhl}c')
    print()

    # Financials
    gross = ss.get("gross_pnl_before_fees") or 0
    fees  = ss.get("fees_paid") or 0
    net   = ss.get("net_pnl")   or 0
    gpf   = ec.get("gross_profit_factor") or 0
    exp   = (ec.get("gross_expectancy_pct") or 0)

    print(f'  FINANCIALS:')
    print(f'    Gross PnL: ${gross:.2f}  |  Fees: ${fees:.2f}  |  Net: ${net:.2f}')
    print(f'    Gross PF:  {gpf:.3f}  |  Gross expectancy/trade: {exp:+.4f}%')
    print()

    # Case verdict
    avg_win  = abs(mwp)
    avg_loss = abs(mlp)
    if biglg and gross != 0:
        biggest_loser_share = abs(biglg) / (abs(gross) + abs(biglg) + 1e-8) * 100
    else:
        biggest_loser_share = 0

    print(f'  CASE VERDICT:')
    if avg_win > 0 and avg_loss > 0:
        rr = avg_win / avg_loss
        print(f'    Avg winner/loser ratio: {rr:.2f}x')
    print(f'    Biggest loser share of gross loss pool: {biggest_loser_share:.1f}%')
    if biggest_loser_share > 40:
        print(f'    => CASE 1: A few giant losers are driving poor PF')
        print(f'       Fix: proportional loss-duration penalty or stop-loss')
    elif gpf < 0.95 and biggest_loser_share < 25:
        print(f'    => CASE 2: Many small random trades, no clear edge')
        print(f'       Fix: better features, more data, different entry criteria')
    else:
        print(f'    => BORDERLINE: mixed evidence, need more data')

print('\n' + '='*65)
