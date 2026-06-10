/**
 * binanceStream.ts  (GOD-5 update)
 * ──────────────────────────────────
 * GOD-5: Integrated UserDataStream for real-time balance/fill updates.
 *        Added getUserDataStream() accessor so ccxtClient and executioner
 *        can query instant balances instead of REST polling.
 */

import ccxt, { binance as Binance, OHLCV } from "ccxt";
import { EventEmitter } from "events";
import { CONFIG, Candle } from "../utils/types";
import { logger }         from "../utils/logger";
import { Tick }           from "../database/mongoSchemas";
import { UserDataStream } from "./userDataStream";

const TF_MS: Record<string, number> = {
  "1m": 60_000, "5m": 300_000, "15m": 900_000,
  "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
};

export class BinanceWatcher extends EventEmitter {
  private exchange:   Binance;
  private buffer:     Candle[]  = [];
  private isRunning:  boolean   = false;
  private reconnects: number    = 0;
  private killSignal: boolean   = false;
  private lastEmittedCandleTs: number | null = null;

  // GOD-5: userData stream for instant balance/fill notifications
  private userData: UserDataStream;

  constructor() {
    super();
    const exchangeClass = this.resolveExchangeClass();
    this.exchange = new exchangeClass({
      apiKey:          CONFIG.apiKey    || undefined,
      secret:          CONFIG.apiSecret || undefined,
      enableRateLimit: true,
      options: {
        defaultType: "spot",
        ...(CONFIG.useTestnet && { urls: { api: { rest: "https://testnet.binance.vision" } } }),
      },
    }) as Binance;

    // GOD-5: create userData stream (starts lazily in start())
    this.userData = new UserDataStream(this.exchange);
  }

  private resolveExchangeClass(): typeof Binance {
    try {
      const pro = require("ccxt/pro");
      logger.info("[Watcher] Using ccxt/pro WebSocket client");
      return pro.binance;
    } catch {
      logger.warn("[Watcher] ccxt/pro not found — falling back to REST polling");
      return ccxt.binance;
    }
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  async start(): Promise<void> {
    if (this.isRunning) return;
    this.isRunning  = true;
    this.killSignal = false;

    logger.info(`[Watcher] Starting | ${CONFIG.pair} | ${CONFIG.timeframe}`);

    await this.prefillBuffer();

    // GOD-5: start userData stream alongside market data stream
    await this.userData.start();

    this.emit("ready");
    this.streamLoop();
  }

  stop(): void {
    logger.info("[Watcher] Stopping");
    this.killSignal = true;
    this.isRunning  = false;
    this.userData.stop();  // GOD-5: clean shutdown
    try { this.exchange.close(); } catch {}
  }

  getBuffer(): Candle[] { return [...this.buffer]; }

  /** GOD-5: accessor for instant balance queries */
  getUserDataStream(): UserDataStream { return this.userData; }

  // ── Buffer Pre-Fill ─────────────────────────────────────────────────────────

  private async prefillBuffer(): Promise<void> {
    const needed = Math.max(CONFIG.windowSize + 50, 300);
    logger.info(`[Watcher] Pre-filling ${needed} candles...`);
    try {
      const raw = await this.exchange.fetchOHLCV(CONFIG.pair, CONFIG.timeframe, undefined, needed);
      const now  = Date.now();
      this.buffer = raw.map(this.rawToCandle).filter(c => this.isClosedCandle(c, now));
      this.lastEmittedCandleTs = this.buffer.at(-1)?.timestamp ?? null;
      logger.info(`[Watcher] Buffer ready: ${this.buffer.length} candles`);
    } catch (err) {
      logger.error("[Watcher] Pre-fill failed", { err }); throw err;
    }
  }

  // ── Stream Loop ─────────────────────────────────────────────────────────────

  private async streamLoop(): Promise<void> {
    while (!this.killSignal) {
      try {
        await this.watchCandles();
      } catch (err: any) {
        if (this.killSignal) break;
        this.reconnects++;
        const backoff = Math.min(1000 * 2 ** this.reconnects, 30_000);
        logger.warn(`[Watcher] Reconnecting in ${backoff}ms (attempt ${this.reconnects})`, { err: err?.message });
        await sleep(backoff);
      }
    }
  }

  private async watchCandles(): Promise<void> {
    if (typeof (this.exchange as any).watchOHLCV === "function") {
      while (!this.killSignal) {
        const candles = await (this.exchange as any).watchOHLCV(CONFIG.pair, CONFIG.timeframe);
        for (const raw of candles) {
          await this.maybeEmitClosedCandle(this.rawToCandle(raw));
        }
        this.reconnects = 0;
      }
      return;
    }

    // REST fallback
    logger.info("[Watcher] REST polling mode");
    while (!this.killSignal) {
      const raw = await this.exchange.fetchOHLCV(CONFIG.pair, CONFIG.timeframe, undefined, 5);
      for (const item of raw) await this.maybeEmitClosedCandle(this.rawToCandle(item));
      const pollMs = (TF_MS[CONFIG.timeframe] ?? 60_000) / 4;  // FIX-4: guard NaN
      await sleep(Math.min(pollMs, 15_000));
    }
  }

  // ── Candle Processing ───────────────────────────────────────────────────────

  private async maybeEmitClosedCandle(candle: Candle): Promise<void> {
    if (!this.isClosedCandle(candle)) return;
    if (this.lastEmittedCandleTs !== null && candle.timestamp <= this.lastEmittedCandleTs) return;
    this.lastEmittedCandleTs = candle.timestamp;
    await this.onClosedCandle(candle);
  }

  private isClosedCandle(candle: Candle, now = Date.now()): boolean {
    return candle.timestamp + (TF_MS[CONFIG.timeframe] ?? 60_000) <= now;
  }

  private async onClosedCandle(candle: Candle): Promise<void> {
    this.buffer.push(candle);
    const maxBuf = Math.max(CONFIG.windowSize + 50, 200);
    if (this.buffer.length > maxBuf) this.buffer.shift();

    logger.debug("[Watcher] Candle", { ts: new Date(candle.timestamp).toISOString(), close: candle.close });

    this.persistCandle(candle).catch(e => logger.warn("[Watcher] DB persist failed", { err: e?.message }));

    if (this.buffer.length >= Math.max(CONFIG.windowSize, 200)) {
      this.emit("candle", [...this.buffer]);
    }
  }

  private async persistCandle(candle: Candle): Promise<void> {
    await Tick.updateOne(
      { pair: CONFIG.pair, timeframe: CONFIG.timeframe, timestamp: new Date(candle.timestamp) },
      { $set: { open: candle.open, high: candle.high, low: candle.low, close: candle.close, volume: candle.volume } },
      { upsert: true }
    );
  }

  // ccxt's OHLCV tuple elements are typed as possibly undefined; live candles
  // always carry values, so default to 0 (a 0-timestamp candle is dropped by
  // the lastEmittedCandleTs dedupe in maybeEmitClosedCandle).
  private rawToCandle(raw: OHLCV): Candle {
    return {
      timestamp: raw[0] ?? 0, open: raw[1] ?? 0, high: raw[2] ?? 0,
      low: raw[3] ?? 0, close: raw[4] ?? 0, volume: raw[5] ?? 0,
    };
  }
}

function sleep(ms: number): Promise<void> { return new Promise(r => setTimeout(r, ms)); }
