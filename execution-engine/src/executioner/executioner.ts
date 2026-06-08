/**
 * executioner.ts  (GOD-2 update)
 * ──────────────────────────────
 * GOD-2: Switched from marketBuy/marketSell to limitBuy/limitSell.
 *        Handles partial fills — if limit cancels before fully filled,
 *        we accept the partial and size the position to actual fill.
 * GOD-3: buildTradeResult now uses CONFIG.effectiveFeeRate (BNB-adjusted).
 */

import { Signal, ModelOutput, Position, TradeResult, OrderSide, ExecutionMode, CONFIG } from "../utils/types";
import { RiskManager }           from "./riskManager";
import { BinanceClient }         from "./ccxtClient";
import { logger }                from "../utils/logger";
import { Trade, Snapshot, SignalLog } from "../database/mongoSchemas";
import { Notifier }              from "../utils/notifier";
import { recoverPositionState }  from "./stateRecovery";

export class Executioner {
  private risk:     RiskManager;
  private client:   BinanceClient;
  private notifier: Notifier;

  private position:        Position | null = null;
  private stopLossOrderId: string | null   = null;
  private paperBalance:    number          = 10_000;
  private isProcessing:    boolean         = false;

  constructor(risk: RiskManager, client: BinanceClient, notifier: Notifier) {
    this.risk = risk; this.client = client; this.notifier = notifier;
  }

  // ── Init ────────────────────────────────────────────────────────────────────

  async initialize(): Promise<void> {
    // GOD-3: detect BNB fee discount before any trading
    await this.client.initialize();

    let balance: number;

    if (CONFIG.executionMode === ExecutionMode.LIVE) {
      balance = await this.client.getUsdtBalance();

      const recovery = await recoverPositionState(
        this.client.rawExchange, CONFIG.stopLossPct, CONFIG.takeProfitPct,
      );
      if (recovery.position) {
        this.position        = recovery.position;
        this.stopLossOrderId = recovery.stopLossOrderId;
        balance = Math.max(0, balance - recovery.position.entryPrice * recovery.position.size);
        await this.notifier.sendAlert(
          `⚠️ Crash recovery: monitoring open position\n` +
          `Entry: $${recovery.position.entryPrice.toFixed(4)} | Size: ${recovery.position.size.toFixed(6)}`
        );
      }
    } else if (CONFIG.executionMode === ExecutionMode.PAPER) {
      balance = this.paperBalance;
      logger.info(`[Exec] PAPER MODE | Virtual balance: $${balance.toFixed(2)}`);
    } else {
      balance = 10_000;
      logger.info("[Exec] MOCK MODE | All orders simulated");
    }

    this.risk.initialize(balance);
    logger.info(`[Exec] Ready | Mode: ${CONFIG.executionMode}`);
  }

  // ── Signal Entry Point ───────────────────────────────────────────────────────

  async onSignal(output: ModelOutput, currentPrice: number): Promise<void> {
    if (this.risk.isKilled) { logger.warn("[Exec] Killed — ignoring signal"); return; }

    if (this.isProcessing) {
      logger.warn("[Exec] Previous signal still processing — skip");
      return;
    }
    this.isProcessing = true;

    try {
      await this.persistSignal(output, currentPrice);

      if (this.position) {
        await this.monitorPosition(currentPrice, output);
      } else if (output.signal === Signal.BUY) {
        await this.tryEnter(output, currentPrice);
      } else {
        logger.debug(`[Exec] ${Signal[output.signal]} (${(output.confidence*100).toFixed(1)}%) — flat`);
      }
    } finally {
      this.isProcessing = false;
    }
  }

  // ── Entry ────────────────────────────────────────────────────────────────────

  private async tryEnter(output: ModelOutput, currentPrice: number): Promise<void> {
    if (output.confidence < CONFIG.minConfidence) {
      logger.info(`[Exec] BUY rejected — conf ${(output.confidence*100).toFixed(1)}% < ${(CONFIG.minConfidence*100).toFixed(0)}%`);
      await this.updateSignalSkipReason("LOW_CONFIDENCE");
      return;
    }

    const size = this.risk.calculatePositionSize(currentPrice);
    if (size === 0) { logger.warn("[Exec] Position size = 0 — skipping"); return; }

    logger.info(`[Exec] 🟢 ENTERING`, { price: currentPrice.toFixed(4), size: size.toFixed(6), conf: (output.confidence*100).toFixed(1)+"%" });

    // GOD-2: use limitBuy instead of marketBuy
    const order = await this.client.limitBuy(size);
    if (!order || order.filledSize <= 0) {
      logger.error("[Exec] Limit order returned zero fill — aborting entry");
      return;
    }

    // GOD-2: handle partial fill — use actual filled size, not requested size
    if (order.isPartial) {
      logger.warn(`[Exec] Partial fill accepted: ${order.filledSize.toFixed(6)} of ${order.size.toFixed(6)}`);
    }

    // SL/TP anchored to actual fill price (not pre-order estimate)
    const actualPrice = order.price > 0 ? order.price : currentPrice;
    const { stopLoss, takeProfit } = this.risk.calculateExitLevels(actualPrice);

    this.position = {
      entryPrice:    actualPrice,
      entryTime:     order.timestamp,
      size:          order.filledSize,   // actual filled size
      stopLoss, takeProfit,
      unrealizedPnl: 0,
    };

    this.risk.updateBalance(this.risk.balance - actualPrice * order.filledSize);

    this.stopLossOrderId = await this.client.placeStopLossOrder(order.filledSize, stopLoss);

    await this.notifier.sendTradeAlert("BUY", {
      pair: CONFIG.pair, price: actualPrice, size: order.filledSize,
      stopLoss, takeProfit, confidence: output.confidence, mode: CONFIG.executionMode,
    });
  }

  // ── Monitor ──────────────────────────────────────────────────────────────────

  private async monitorPosition(currentPrice: number, output: ModelOutput): Promise<void> {
    if (!this.position) return;
    this.position.unrealizedPnl = (currentPrice - this.position.entryPrice) / this.position.entryPrice;

    const exitReason = this.risk.checkExitConditions(this.position, currentPrice);
    if (exitReason) { await this.closePosition(currentPrice, exitReason); return; }

    if (output.signal === Signal.SELL && output.confidence >= CONFIG.minConfidence) {
      logger.info("[Exec] AI SELL signal — closing");
      await this.closePosition(currentPrice, "SIGNAL");
    }
  }

  // ── Exit ─────────────────────────────────────────────────────────────────────

  private async closePosition(exitPrice: number, exitReason: TradeResult["exitReason"]): Promise<void> {
    if (!this.position) return;
    const closed = { ...this.position };

    logger.info(`[Exec] 🔴 CLOSING`, { price: exitPrice.toFixed(4), reason: exitReason, pnl: (closed.unrealizedPnl*100).toFixed(3)+"%" });

    if (this.stopLossOrderId) { await this.client.cancelOrder(this.stopLossOrderId); this.stopLossOrderId = null; }

    // GOD-2: use limitSell (market sell only for kill switch emergencies)
    const order  = exitReason === "KILL_SWITCH"
      ? await this.client.marketSell(closed.size)
      : await this.client.limitSell(closed.size);

    // GOD-3: pass effective fee rate for accurate PnL calculation
    const result = this.risk.buildTradeResult(
      closed, order.price, exitReason, CONFIG.effectiveFeeRate ?? 0.001
    );

    const newBalance = this.risk.balance + order.price * order.filledSize - order.fee;
    this.risk.updateBalance(newBalance);
    if (CONFIG.executionMode === ExecutionMode.PAPER) this.paperBalance = newBalance;

    this.position = null;

    await this.persistTrade(result, closed, order.orderId);
    await this.saveSnapshot();
    await this.notifier.sendTradeAlert("SELL", {
      pair: CONFIG.pair, price: order.price, pnlPct: result.pnlPct,
      pnlUsdt: result.pnlUsdt, exitReason, mode: CONFIG.executionMode, newBalance: this.risk.balance,
    });
  }

  // ── Kill Switch ──────────────────────────────────────────────────────────────

  async emergencyStop(reason = "Manual kill switch"): Promise<void> {
    logger.error(`[Exec] 🚨 EMERGENCY STOP: ${reason}`);
    this.risk.kill(reason);
    await this.client.cancelAllOrders();
    if (this.position) {
      const price = await this.client.getCurrentPrice();
      await this.closePosition(price, "KILL_SWITCH");
    }
    await this.notifier.sendAlert(`🚨 EMERGENCY STOP\n${reason}`);
  }

  get currentPosition(): Position | null { return this.position; }
  get sessionSummary() { return this.risk.getSessionSummary(); }

  // ── DB ────────────────────────────────────────────────────────────────────────

  private async persistTrade(result: TradeResult, pos: Position, orderId?: string): Promise<void> {
    try {
      await Trade.create({
        pair: CONFIG.pair, mode: CONFIG.executionMode, side: result.side,
        entryPrice: result.entryPrice, exitPrice: result.exitPrice,
        size: result.size, pnlPct: result.pnlPct, pnlUsdt: result.pnlUsdt,
        feePaid: result.feePaid, stopLoss: pos.stopLoss, takeProfit: pos.takeProfit,
        entryTime: new Date(Date.now() - result.durationMs), exitTime: new Date(),
        durationMs: result.durationMs, exitReason: result.exitReason, orderId,
      });
    } catch (e: any) { logger.warn("[Exec] Trade persist failed", { err: e.message }); }
  }

  private async persistSignal(output: ModelOutput, price: number): Promise<void> {
    try {
      await SignalLog.create({
        pair: CONFIG.pair, timestamp: new Date(), signal: output.signal,
        probHold: output.probHold, probBuy: output.probBuy, probSell: output.probSell,
        confidence: output.confidence, acted: false,
      });
    } catch {}
  }

  private async updateSignalSkipReason(reason: string): Promise<void> {
    try {
      await SignalLog.findOneAndUpdate(
        { pair: CONFIG.pair }, { $set: { skipReason: reason } }, { sort: { timestamp: -1 } }
      );
    } catch {}
  }

  private async saveSnapshot(): Promise<void> {
    try {
      const s = this.risk.getSessionSummary();
      await Snapshot.create({
        timestamp: new Date(), mode: CONFIG.executionMode,
        totalBalance: s.currentBalance,
        availableUsdt: this.position ? 0 : s.currentBalance,
        positionValue: this.position ? this.position.size * this.position.entryPrice : 0,
        totalPnlPct: s.sessionPnlPct, openPosition: !!this.position,
      });
    } catch {}
  }
}
