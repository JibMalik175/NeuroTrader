/**
 * userDataStream.ts  (GOD-5)
 * ───────────────────────────
 * Subscribes to Binance's private !userData WebSocket stream.
 *
 * BEFORE: After any fill, the bot called fetchBalance() via REST.
 *         This takes 200-500ms and may return stale data if called
 *         immediately after a fill.
 *
 * AFTER:  Binance pushes outboundAccountPosition within 50ms of any
 *         fill. Balance is always current for the next decision cycle.
 *         Order fills are cached so waitForFill() resolves instantly
 *         instead of polling every second.
 *
 * Events emitted:
 *   "balance"    → Map<string, BalanceEntry>  on any balance change
 *   "orderFill"  → { orderId, price, filledSize, status }
 *
 * Binance listenKey expires after 60 minutes — renewed every 30 min.
 * If the stream fails, falls back to REST polling gracefully.
 */

import { EventEmitter } from "events";
import ccxt             from "ccxt";
import WebSocket        from "ws";
import { CONFIG, ExecutionMode } from "../utils/types";
import { logger }                from "../utils/logger";

export interface BalanceEntry {
  asset:     string;
  free:      number;
  locked:    number;
  updatedAt: number;
}

export interface OrderFillEvent {
  orderId:    string;
  price:      number;
  filledSize: number;
  status:     string;
}

export class UserDataStream extends EventEmitter {
  private exchange:     ccxt.binance;
  private listenKey:    string | null = null;
  private ws:           WebSocket | null = null;
  private renewTimer:   ReturnType<typeof setInterval> | null = null;
  private balances      = new Map<string, BalanceEntry>();
  private orderFills    = new Map<string, OrderFillEvent>();
  private active        = false;

  constructor(exchange: ccxt.binance) {
    super();
    this.exchange = exchange;
  }

  // ── Start / Stop ───────────────────────────────────────────────────────────

  async start(): Promise<void> {
    if (CONFIG.executionMode !== ExecutionMode.LIVE) {
      logger.info("[UserData] Non-LIVE mode — userData stream skipped");
      return;
    }

    try {
      await this.createListenKey();
      this.connectWebSocket();
      this.scheduleRenewal();
      this.active = true;
      logger.info("[UserData] ✅ Stream active — instant balance updates enabled");
    } catch (err: any) {
      logger.warn(`[UserData] Failed to start: ${err.message} — REST polling fallback`);
    }
  }

  stop(): void {
    this.active = false;
    if (this.renewTimer) clearInterval(this.renewTimer);
    if (this.ws) { try { this.ws.close(); } catch {} }
    if (this.listenKey) {
      // Best-effort stream close notification to Binance
      (this.exchange as any).sapiDeleteUserdataStream?.({ listenKey: this.listenKey }).catch(() => {});
    }
    logger.info("[UserData] Stream stopped");
  }

  // ── Balance API ───────────────────────────────────────────────────────────

  /**
   * Returns the free balance for an asset.
   * Updated within 50ms of any fill via userData stream.
   * Falls back to REST if no cached value or cache is stale (>5s).
   */
  async getFreeBalance(asset: string): Promise<number> {
    const cached = this.balances.get(asset);
    if (cached && Date.now() - cached.updatedAt < 5_000) return cached.free;

    try {
      const bal  = await this.exchange.fetchBalance();
      const free = bal.free?.[asset] ?? 0;
      this.balances.set(asset, { asset, free, locked: bal.used?.[asset] ?? 0, updatedAt: Date.now() });
      return free;
    } catch (err: any) {
      logger.warn(`[UserData] Balance fallback failed: ${err.message}`);
      return cached?.free ?? 0;
    }
  }

  /**
   * Returns the fill event for an order if received via userData stream.
   * Allows waitForFill() to resolve instantly rather than polling REST.
   */
  getOrderFill(orderId: string): OrderFillEvent | undefined {
    return this.orderFills.get(orderId);
  }

  // ── ListenKey Management ──────────────────────────────────────────────────

  private async createListenKey(): Promise<void> {
    const res      = await (this.exchange as any).sapiPostUserdataStream();
    this.listenKey = res.listenKey;
    logger.debug(`[UserData] ListenKey: ${this.listenKey?.substring(0, 16)}...`);
  }

  private async renewListenKey(): Promise<void> {
    if (!this.listenKey) return;
    try {
      await (this.exchange as any).sapiPutUserdataStream({ listenKey: this.listenKey });
      logger.debug("[UserData] ListenKey renewed");
    } catch (err: any) {
      logger.warn(`[UserData] Renewal failed — recreating: ${err.message}`);
      await this.createListenKey();
    }
  }

  private scheduleRenewal(): void {
    // Binance listenKey expires after 60 minutes — renew every 30 to be safe
    this.renewTimer = setInterval(() => {
      this.renewListenKey().catch(e => logger.warn(`[UserData] Renewal error: ${e.message}`));
    }, 30 * 60 * 1000);
  }

  // ── WebSocket ─────────────────────────────────────────────────────────────

  private connectWebSocket(): void {
    if (!this.listenKey) return;

    const url = CONFIG.useTestnet
      ? `wss://testnet.binance.vision/ws/${this.listenKey}`
      : `wss://stream.binance.com:9443/ws/${this.listenKey}`;

    this.ws = new WebSocket(url);

    this.ws.on("open",    () => logger.debug("[UserData] WebSocket connected"));
    this.ws.on("message", (raw: Buffer) => {
      try { this.handleEvent(JSON.parse(raw.toString())); }
      catch (e: any) { logger.debug(`[UserData] Parse error: ${e.message}`); }
    });
    this.ws.on("error", (err: Error) => {
      logger.warn(`[UserData] WebSocket error: ${err.message}`);
    });
    this.ws.on("close", () => {
      if (!this.active) return;
      logger.warn("[UserData] WebSocket closed — reconnecting in 5s");
      setTimeout(() => this.connectWebSocket(), 5_000);
    });
  }

  // ── Event Handlers ────────────────────────────────────────────────────────

  private handleEvent(event: any): void {
    switch (event.e) {

      // Fired after any fill — provides updated balances for all affected assets
      case "outboundAccountPosition": {
        for (const b of (event.B ?? [])) {
          this.balances.set(b.a, {
            asset: b.a, free: parseFloat(b.f),
            locked: parseFloat(b.l), updatedAt: Date.now(),
          });
        }
        this.emit("balance", this.balances);
        logger.debug(`[UserData] Balances updated: ${(event.B ?? []).map((b: any) => b.a).join(", ")}`);
        break;
      }

      // Order execution report — fill, partial fill, cancel
      case "executionReport": {
        const orderId    = String(event.i);
        const status     = event.X as string;
        const filledQty  = parseFloat(event.z ?? "0");
        const avgPrice   = parseFloat(event.ap ?? event.p ?? "0");

        if (status === "FILLED" || status === "PARTIALLY_FILLED") {
          const fill: OrderFillEvent = { orderId, price: avgPrice, filledSize: filledQty, status };
          this.orderFills.set(orderId, fill);
          this.emit("orderFill", fill);
          logger.debug(`[UserData] ${status} | order=${orderId} qty=${filledQty} @ $${avgPrice}`);
        }
        break;
      }
    }
  }
}
