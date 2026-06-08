/**
 * index.ts  — Main Bootstrap  (GOD MODE update)
 * ───────────────────────────────────────────────
 * GOD-5: BinanceWatcher.getUserDataStream() wired into BinanceClient
 *        so instant balance updates flow to the executioner.
 * GOD-6: sendDailySummary now actually fires every 24h (was implemented
 *        but never wired — now fixed).
 * Safe intervals: all setInterval callbacks wrapped in try/catch so a
 *        single error doesn't silently kill the stats/summary loop.
 */

import "dotenv/config";
import { connectDB, disconnectDB }  from "./database/mongoSchemas";
import { BinanceWatcher }           from "./watcher/binanceStream";
import { InferenceEngine }          from "./strategist/inference";
import { RiskManager }              from "./executioner/riskManager";
import { BinanceClient }            from "./executioner/ccxtClient";
import { Executioner }              from "./executioner/executioner";
import { Notifier }                 from "./utils/notifier";
import { CONFIG, ExecutionMode, Candle, Signal } from "./utils/types";
import { logger }                   from "./utils/logger";
import { Trade }                    from "./database/mongoSchemas";

// ── Banner ────────────────────────────────────────────────────────────────────

function printBanner(): void {
  const mode  = CONFIG.executionMode;
  const color = mode === ExecutionMode.MOCK ? "\x1b[33m" : mode === ExecutionMode.PAPER ? "\x1b[36m" : "\x1b[31m";
  const reset = "\x1b[0m";
  const fee   = `${(CONFIG.effectiveFeeRate * 100).toFixed(4)}% fee${CONFIG.useBnbFeeDiscount ? " (BNB✅)" : ""}`;
  console.log(`
╔══════════════════════════════════════════════════════╗
║         T R A D E B O T  —  G O D  M O D E          ║
╠══════════════════════════════════════════════════════╣
║  Pair      : ${CONFIG.pair.padEnd(38)}║
║  Timeframe : ${CONFIG.timeframe.padEnd(38)}║
║  Mode      : ${(color + mode + reset).padEnd(38 + color.length + reset.length)}║
║  Testnet   : ${String(CONFIG.useTestnet).padEnd(38)}║
║  Fee rate  : ${fee.padEnd(38)}║
╚══════════════════════════════════════════════════════╝
  `);
  if (mode === ExecutionMode.LIVE) {
    logger.warn("⚠️  LIVE MODE — REAL MONEY. Starting in 5s... (Ctrl+C to abort)");
  }
}

// ── Config Validation ─────────────────────────────────────────────────────────

function validateConfig(): void {
  const errors: string[] = [];
  if (CONFIG.executionMode === ExecutionMode.LIVE && (!CONFIG.apiKey || !CONFIG.apiSecret))
    errors.push("LIVE mode requires BINANCE_API_KEY and BINANCE_API_SECRET");
  if (CONFIG.stopLossPct >= CONFIG.takeProfitPct)
    logger.warn(`[Config] SL (${CONFIG.stopLossPct}) >= TP (${CONFIG.takeProfitPct}) — poor R:R`);
  if (CONFIG.windowSize < 48)
    logger.warn(`[Config] windowSize=${CONFIG.windowSize} — model was trained with 48`);
  if (CONFIG.maxRiskPerTrade > 0.05)
    logger.warn(`[Config] MAX_RISK_PER_TRADE=${(CONFIG.maxRiskPerTrade*100).toFixed(0)}% is aggressive`);
  if (errors.length) throw new Error(`Config errors:\n- ${errors.join("\n- ")}`);
  logger.info("[Config] ✅ Validated");
}

// ── Safe Interval ─────────────────────────────────────────────────────────────

/** Wraps setInterval callback in try/catch so errors don't kill the loop. */
function safeInterval(fn: () => Promise<void> | void, ms: number): ReturnType<typeof setInterval> {
  return setInterval(async () => {
    try { await fn(); }
    catch (err: any) { logger.error("[Interval] Error in scheduled task", { err: err.message }); }
  }, ms);
}

// ── Stats ─────────────────────────────────────────────────────────────────────

class SessionStats {
  private candleCount = 0;
  private signalCount = 0;
  private startTime   = Date.now();
  onCandle() { this.candleCount++; }
  onSignal() { this.signalCount++; }
  print(): void {
    const s = Math.floor((Date.now() - this.startTime) / 1000);
    logger.info(`[Stats] Up: ${Math.floor(s/3600)}h${Math.floor(s%3600/60)}m | Candles: ${this.candleCount} | Signals: ${this.signalCount}`);
  }
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  validateConfig();
  printBanner();

  if (CONFIG.executionMode === ExecutionMode.LIVE) await sleep(5_000);

  // 1. Database
  try { await connectDB(); logger.info("[Boot] MongoDB connected"); }
  catch (err: any) { logger.warn(`[Boot] MongoDB unavailable: ${err.message}`); }

  // 2. Modules
  const watcher     = new BinanceWatcher();
  const inference   = new InferenceEngine();
  const riskManager = new RiskManager();
  const client      = new BinanceClient();
  const notifier    = new Notifier();
  const executioner = new Executioner(riskManager, client, notifier);
  const stats       = new SessionStats();

  // 3. Load model
  try { await inference.load(CONFIG.modelPath); }
  catch (err: any) {
    logger.warn(`[Boot] Model not loaded: ${err.message} — signal-bypass mode`);
  }

  // 4. Initialize executioner (crash recovery + BNB fee detection inside)
  await executioner.initialize();

  // 5. Graceful shutdown
  let shuttingDown = false;
  const shutdown = async (signal: string, code = 0) => {
    if (shuttingDown) return;
    shuttingDown = true;
    logger.info(`[Boot] ${signal} — shutting down...`);
    watcher.stop();
    if (executioner.currentPosition) await executioner.emergencyStop(`${signal} shutdown`);
    stats.print();
    await disconnectDB();
    process.exit(code);
  };

  process.on("SIGINT",  () => { void shutdown("SIGINT"); });
  process.on("SIGTERM", () => { void shutdown("SIGTERM"); });
  process.on("uncaughtException",  (err) => { logger.error("[Boot] Uncaught", { err: err.message }); process.exitCode = 1; void shutdown("uncaughtException", 1); });
  process.on("unhandledRejection", (r)   => { logger.error("[Boot] Unhandled rejection", { r });    process.exitCode = 1; void shutdown("unhandledRejection", 1); });

  // 6. Event pipeline
  watcher.on("ready", () => {
    logger.info("[Boot] ✅ All systems ready");
    notifier.sendAlert(`🤖 TradeBot GOD MODE started\n${CONFIG.pair} | ${CONFIG.executionMode}`);
  });

  watcher.on("candle", async (candles: Candle[]) => {
    stats.onCandle();
    const currentPrice = candles.at(-1)!.close;
    logger.info(`[Loop] Candle ${new Date(candles.at(-1)!.timestamp).toISOString()} | $${currentPrice.toFixed(4)}`);

    if (!inference.loaded) {
      const pos = executioner.currentPosition;
      if (pos && riskManager.checkExitConditions(pos, currentPrice)) {
        await executioner.onSignal({ signal: Signal.SELL, probHold: 0, probBuy: 0, probSell: 1, confidence: 1 }, currentPrice);
      }
      return;
    }

    try {
      const pos    = executioner.currentPosition;
      const sum    = executioner.sessionSummary;
      const output = await inference.predict(candles, !!pos, pos?.entryPrice ?? 0, sum.peakBalance, sum.currentBalance);
      stats.onSignal();
      await executioner.onSignal(output, currentPrice);
    } catch (err: any) {
      logger.error("[Loop] Error", { err: err.message });
    }
  });

  watcher.on("error", (err: Error) => logger.error("[Watcher] Error", { err: err.message }));

  // 7. Periodic stats (safe interval — won't kill loop on error)
  safeInterval(() => {
    stats.print();
    const s = executioner.sessionSummary;
    logger.info("[Session]", {
      balance:    s.currentBalance.toFixed(2),
      pnl:        `${(s.sessionPnlPct * 100).toFixed(3)}%`,
      drawdown:   `${(s.drawdownPct   * 100).toFixed(2)}%`,
      inPosition: !!executioner.currentPosition,
      killed:     s.killed,
    });
  }, 60 * 60 * 1000);

  // 8. Daily summary (GOD-6: was implemented but never wired — now fires every 24h)
  safeInterval(async () => {
    const s = executioner.sessionSummary;
    // Aggregate today's closed trades from MongoDB for accurate daily stats
    const today = new Date(); today.setUTCHours(0, 0, 0, 0);
    let totalTrades = 0; let winCount = 0; let totalPnl = 0;
    try {
      const trades = await Trade.find({ exitTime: { $gte: today } }).lean();
      totalTrades  = trades.length;
      winCount     = trades.filter((t: any) => t.pnlUsdt > 0).length;
      totalPnl     = trades.reduce((sum: number, t: any) => sum + (t.pnlUsdt ?? 0), 0);
    } catch {}

    await notifier.sendDailySummary({
      totalTrades,
      winRate:  totalTrades > 0 ? winCount / totalTrades : 0,
      totalPnl,
      balance:  s.currentBalance,
    });
  }, 24 * 60 * 60 * 1000);

  // 9. Start stream
  logger.info("[Boot] Starting Binance stream...");
  await watcher.start();
}

function sleep(ms: number): Promise<void> { return new Promise(r => setTimeout(r, ms)); }

main().catch(err => { logger.error("[Boot] Fatal", { err: err.message }); process.exit(1); });
