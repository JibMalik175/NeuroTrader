/**
 * stateRecovery.ts
 * ────────────────
 * FIX #1: Crash Recovery
 *
 * When the Node.js process restarts mid-trade, position is null.
 * This module queries Binance directly to reconstruct the position
 * state so the bot can resume monitoring without opening a duplicate
 * trade or orphaning a live position.
 *
 * Strategy:
 *   1. Fetch open orders  → find any resting SL order (proves we were in a trade)
 *   2. Fetch recent fills → find the last BUY fill to recover entryPrice + size
 *   3. Re-calculate SL/TP from the recovered entry price
 *   4. Return a fully reconstructed Position object (or null if flat)
 */

import ccxt from "ccxt";
import { Position, CONFIG, ExecutionMode } from "../utils/types";
import { logger } from "../utils/logger";

export interface RecoveryResult {
  position:       Position | null;
  stopLossOrderId: string | null;
}

export async function recoverPositionState(
  exchange: ccxt.binance,
  stopLossPct:   number,
  takeProfitPct: number,
): Promise<RecoveryResult> {

  // Only attempt recovery in LIVE mode — PAPER/MOCK always start fresh
  if (CONFIG.executionMode !== ExecutionMode.LIVE) {
    return { position: null, stopLossOrderId: null };
  }

  logger.info("[Recovery] Checking Binance for orphaned positions...");

  try {
    // ── Step 1: Check open orders for a resting stop-loss ─────────────────
    const openOrders = await exchange.fetchOpenOrders(CONFIG.pair);

    // A stop-loss-limit SELL order is the fingerprint of an active position
    const slOrder = openOrders.find(o =>
      o.side === "sell" &&
      (o.type === "stop_loss_limit" || o.type === "stop_loss" || o.type === "limit")
    );

    // ── Step 2: Find the last BUY fill to reconstruct entry details ────────
    // Look back up to 50 recent trades to find the fill
    const recentTrades = await exchange.fetchMyTrades(CONFIG.pair, undefined, 50);

    // Walk backwards — find the most recent BUY that hasn't been offset by a SELL
    let recoveredBuySize  = 0;
    let recoveredBuyPrice = 0;
    let recoveredBuyTime  = 0;

    // Count net position: sum of buys minus sum of sells (in base currency)
    let netSize = 0;
    let weightedEntryPrice = 0;

    // Iterate oldest→newest so we can accumulate correctly
    for (const trade of recentTrades.sort((a, b) => a.timestamp - b.timestamp)) {
      if (trade.side === "buy") {
        // Accumulate into position
        const prevValue    = weightedEntryPrice * netSize;
        const tradeValue   = (trade.price) * (trade.amount);
        netSize           += trade.amount;
        weightedEntryPrice = netSize > 0 ? (prevValue + tradeValue) / netSize : 0;
        recoveredBuyTime   = trade.timestamp;
      } else if (trade.side === "sell") {
        // Position was reduced or closed
        netSize -= trade.amount;
        if (netSize <= 0) {
          // Position was fully closed — reset
          netSize = 0;
          weightedEntryPrice = 0;
        }
      }
    }

    // ── Step 3: Determine if we're actually in a position ─────────────────
    // Binance minimum tradeable unit for BTC is 0.00001 — use that as floor
    const MIN_SIZE = 0.00001;

    if (netSize < MIN_SIZE && !slOrder) {
      logger.info("[Recovery] No open position detected — starting fresh");
      return { position: null, stopLossOrderId: null };
    }

    // If we found net exposure but no SL order, the SL may have been filled
    // (meaning we were stopped out) — treat as flat
    if (netSize < MIN_SIZE && slOrder) {
      logger.warn("[Recovery] Found open SL order but no net buy position — cancelling orphaned order");
      try { await exchange.cancelOrder(slOrder.id, CONFIG.pair); } catch {}
      return { position: null, stopLossOrderId: null };
    }

    // ── Step 4: Reconstruct the Position object ────────────────────────────
    const entryPrice  = weightedEntryPrice;
    const stopLoss    = entryPrice * (1 - stopLossPct);
    const takeProfit  = entryPrice * (1 + takeProfitPct);

    const position: Position = {
      entryPrice,
      entryTime:     recoveredBuyTime || Date.now(),
      size:          netSize,
      stopLoss,
      takeProfit,
      unrealizedPnl: 0,     // will be updated on next candle tick
    };

    logger.warn("[Recovery] ⚠️  RECOVERED ORPHANED POSITION", {
      entryPrice:  entryPrice.toFixed(4),
      size:        netSize.toFixed(6),
      stopLoss:    stopLoss.toFixed(4),
      takeProfit:  takeProfit.toFixed(4),
      slOrderId:   slOrder?.id ?? "none (monitoring only)",
    });

    return {
      position,
      stopLossOrderId: slOrder?.id ?? null,
    };

  } catch (err: any) {
    // Recovery failure must NOT crash the bot — log and start fresh
    logger.error("[Recovery] State recovery failed — starting flat", { err: err.message });
    return { position: null, stopLossOrderId: null };
  }
}
