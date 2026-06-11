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

import type { binance as Binance } from "ccxt";
import { Position, CONFIG, ExecutionMode } from "../utils/types";
import { logger } from "../utils/logger";

export interface RecoveryResult {
  position:       Position | null;
  stopLossOrderId: string | null;
}

export async function recoverPositionState(
  exchange: Binance,
  stopLossPct:   number,
  takeProfitPct: number,
): Promise<RecoveryResult> {

  // Only attempt recovery in LIVE mode — PAPER/MOCK always start fresh
  if (CONFIG.executionMode !== ExecutionMode.LIVE) {
    return { position: null, stopLossOrderId: null };
  }

  logger.info("[Recovery] Checking Binance for orphaned positions...");

  try {
    // ── Step 1: Check open orders for a resting BOT stop-loss ─────────────
    const openOrders = await exchange.fetchOpenOrders(CONFIG.pair);

    // Ownership rule: the bot only claims a position guarded by a stop-loss
    // carrying ITS clientOrderId prefix. Without this filter, recovery summed
    // ALL account fills — the user's manual BTC buys would be adopted as a
    // "bot position" and then managed/SOLD by the bot. A human trading on
    // the same account must never have their coins claimed.
    const isBotOrder = (o: any) =>
      String(o.clientOrderId ?? o.info?.clientOrderId ?? "")
        .startsWith(CONFIG.botOrderPrefix);

    const slOrder = openOrders.find(o =>
      o.side === "sell" &&
      (o.type === "stop_loss_limit" || o.type === "stop_loss" || o.type === "limit") &&
      isBotOrder(o)
    );

    if (openOrders.some(o => o.side === "sell" &&
        (o.type === "stop_loss_limit" || o.type === "stop_loss") && !isBotOrder(o))) {
      logger.warn("[Recovery] Found a stop-loss WITHOUT the bot's tag — treating it " +
                  "as the user's own order, leaving it alone");
    }

    if (!slOrder) {
      logger.info("[Recovery] No bot-tagged stop-loss resting — any net BTC on this " +
                  "account is treated as the USER's holdings, not a bot position");
      return { position: null, stopLossOrderId: null };
    }

    // ── Step 2: Reconstruct the position FROM THE BOT'S OWN SL ORDER ───────
    // The old code summed the last 50 account fills — which included the
    // USER's manual trades, polluting both size and entry price. The tagged
    // SL order is a cleaner source of truth: the bot sized it to its position
    // (size = remaining amount) and priced it at entry × (1 − stopLossPct)
    // (entry = stopPrice / (1 − stopLossPct)). Fully ownership-scoped.
    const MIN_SIZE = 0.00001;  // Binance minimum BTC unit

    const size = slOrder.remaining ?? slOrder.amount ?? 0;
    const stopPrice = Number(
      slOrder.stopPrice ?? (slOrder as any).info?.stopPrice ?? slOrder.price ?? 0
    );

    if (size < MIN_SIZE || stopPrice <= 0) {
      logger.warn("[Recovery] Bot SL order is malformed/dust — cancelling and starting flat");
      try { await exchange.cancelOrder(slOrder.id, CONFIG.pair); } catch {}
      return { position: null, stopLossOrderId: null };
    }

    const entryPrice = stopPrice / (1 - stopLossPct);
    const takeProfit = entryPrice * (1 + takeProfitPct);

    const position: Position = {
      entryPrice,
      entryTime:     slOrder.timestamp ?? Date.now(),
      size,
      stopLoss:      stopPrice,
      takeProfit,
      unrealizedPnl: 0,     // will be updated on next candle tick
    };

    logger.warn("[Recovery] ⚠️  RECOVERED BOT POSITION (from tagged SL)", {
      entryPrice: entryPrice.toFixed(4),
      size:       size.toFixed(6),
      stopLoss:   stopPrice.toFixed(4),
      takeProfit: takeProfit.toFixed(4),
      slOrderId:  slOrder.id,
    });

    return {
      position,
      stopLossOrderId: slOrder.id,
    };

  } catch (err: any) {
    // Recovery failure must NOT crash the bot — log and start fresh
    logger.error("[Recovery] State recovery failed — starting flat", { err: err.message });
    return { position: null, stopLossOrderId: null };
  }
}
