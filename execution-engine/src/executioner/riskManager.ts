/**
 * riskManager.ts  (GOD-3 update)
 * ──────────────────────────────
 * GOD-3: buildTradeResult now reads CONFIG.effectiveFeeRate (BNB-adjusted).
 *        Previously hardcoded 0.001 regardless of BNB discount.
 *        If live rate is 0.075% and model was trained at 0.1%, PnL was
 *        being under-reported by ~25% of fee costs on every trade.
 *
 * IMP-5: Circuit breaker resets daily at UTC midnight (not session-based).
 */

import { CONFIG, Position, TradeResult, OrderSide } from "../utils/types";
import { logger } from "../utils/logger";

const MAX_DAILY_LOSS_PCT = 0.05;

export class RiskManager {
  private killed:           boolean = false;
  private sessionStartBal:  number  = 0;
  private dailyStartBal:    number  = 0;
  private peakBalance:      number  = 0;
  private currentBalance:   number  = 0;
  private dailyLossTripped: boolean = false;
  private dailyResetTimer:  ReturnType<typeof setTimeout> | null = null;

  initialize(initialBalance: number): void {
    this.sessionStartBal  = initialBalance;
    this.dailyStartBal    = initialBalance;
    this.peakBalance      = initialBalance;
    this.currentBalance   = initialBalance;
    this.killed           = false;
    this.dailyLossTripped = false;
    this.scheduleDailyReset();
    logger.info(`[RiskMgr] Initialized | Balance: $${initialBalance.toFixed(2)}`);
  }

  kill(reason: string): void {
    this.killed = true;
    if (this.dailyResetTimer) clearTimeout(this.dailyResetTimer);
    logger.error(`[RiskMgr] 🔴 KILL: ${reason}`);
  }

  get isKilled(): boolean { return this.killed; }

  updateBalance(newBalance: number): void {
    this.currentBalance = newBalance;
    if (newBalance > this.peakBalance) this.peakBalance = newBalance;
    this.checkCircuitBreaker();
  }

  get peak(): number    { return this.peakBalance; }
  get balance(): number { return this.currentBalance; }

  calculatePositionSize(entryPrice: number): number {
    if (this.killed || this.dailyLossTripped) return 0;
    const riskAmount = this.currentBalance * CONFIG.maxRiskPerTrade;
    const slDistance = entryPrice * CONFIG.stopLossPct;
    const size       = riskAmount / slDistance;
    const notional   = size * entryPrice;
    if (notional < 5.5) {
      logger.warn(`[RiskMgr] Notional too small: $${notional.toFixed(2)} < $5.5 min`);
      return 0;
    }
    logger.info(`[RiskMgr] Size: ${size.toFixed(6)} BTC ($${notional.toFixed(2)}) | risk $${riskAmount.toFixed(2)}`);
    return size;
  }

  calculateExitLevels(entryPrice: number): { stopLoss: number; takeProfit: number } {
    const stopLoss   = entryPrice * (1 - CONFIG.stopLossPct);
    const takeProfit = entryPrice * (1 + CONFIG.takeProfitPct);
    logger.info(`[RiskMgr] SL: $${stopLoss.toFixed(4)} | TP: $${takeProfit.toFixed(4)} | R:R ${(CONFIG.takeProfitPct / CONFIG.stopLossPct).toFixed(1)}:1`);
    return { stopLoss, takeProfit };
  }

  checkExitConditions(position: Position, currentPrice: number): "TAKE_PROFIT" | "STOP_LOSS" | null {
    if (currentPrice >= position.takeProfit) {
      logger.info(`[RiskMgr] ✅ TP HIT | ${currentPrice.toFixed(4)} >= ${position.takeProfit.toFixed(4)}`);
      return "TAKE_PROFIT";
    }
    if (currentPrice <= position.stopLoss) {
      logger.warn(`[RiskMgr] 🛑 SL HIT | ${currentPrice.toFixed(4)} <= ${position.stopLoss.toFixed(4)}`);
      return "STOP_LOSS";
    }
    return null;
  }

  /**
   * GOD-3: feeRate parameter now comes from CONFIG.effectiveFeeRate.
   * Default kept at 0.001 for backward compatibility if caller doesn't pass it.
   */
  buildTradeResult(
    position:   Position,
    exitPrice:  number,
    exitReason: TradeResult["exitReason"],
    feeRate:    number = CONFIG.effectiveFeeRate ?? 0.001,  // GOD-3
  ): TradeResult {
    const entryFee = position.size * position.entryPrice * feeRate;
    const exitFee  = position.size * exitPrice           * feeRate;
    const totalFee = entryFee + exitFee;
    const grossPnl = (exitPrice - position.entryPrice) * position.size;
    const netPnl   = grossPnl - totalFee;
    const pnlPct   = netPnl / (position.size * position.entryPrice);

    logger.info(`[RiskMgr] ${netPnl >= 0 ? "✅" : "❌"} Trade | pnl=${( pnlPct*100).toFixed(3)}% | net=$${netPnl.toFixed(4)} | fee=$${totalFee.toFixed(4)} | ${exitReason}`);

    return {
      side: OrderSide.BUY,
      entryPrice: position.entryPrice, exitPrice,
      size: position.size, pnlPct, pnlUsdt: netPnl,
      feePaid: totalFee, durationMs: Date.now() - position.entryTime, exitReason,
    };
  }

  private checkCircuitBreaker(): void {
    if (this.dailyLossTripped || this.killed) return;
    // IMP-5: compare against dailyStartBal (resets at midnight), not sessionStartBal
    const drawdown = (this.dailyStartBal - this.currentBalance) / this.dailyStartBal;
    if (drawdown >= MAX_DAILY_LOSS_PCT) {
      this.dailyLossTripped = true;
      logger.error(`[RiskMgr] ⚠️ CIRCUIT BREAKER — daily loss ${(drawdown*100).toFixed(2)}% | halted until midnight`);
    }
  }

  private scheduleDailyReset(): void {
    if (this.dailyResetTimer) clearTimeout(this.dailyResetTimer);
    const now      = new Date();
    const midnight = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1);
    const delay    = Math.max(midnight - now.getTime(), 60_000);
    this.dailyResetTimer = setTimeout(() => {
      this.dailyStartBal    = this.currentBalance;
      this.peakBalance      = this.currentBalance;
      this.dailyLossTripped = false;
      logger.info(`[RiskMgr] 🔄 Daily reset at UTC midnight | bal=$${this.currentBalance.toFixed(2)}`);
      this.scheduleDailyReset();
    }, delay);
    logger.info(`[RiskMgr] Circuit breaker resets in ${(delay/3600000).toFixed(1)}h`);
  }

  getSessionSummary() {
    return {
      startBalance:   this.sessionStartBal,
      currentBalance: this.currentBalance,
      peakBalance:    this.peakBalance,
      sessionPnlUsdt: this.currentBalance - this.sessionStartBal,
      sessionPnlPct:  (this.currentBalance - this.sessionStartBal) / this.sessionStartBal,
      drawdownPct:    (this.peakBalance - this.currentBalance) / Math.max(this.peakBalance, 1),
      killed:         this.killed,
      circuitBroken:  this.dailyLossTripped,
    };
  }
}
