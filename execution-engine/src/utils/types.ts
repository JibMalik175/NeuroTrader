/**
 * types.ts — Central Type Definitions
 * ──────────────────────────────────────
 * GOD MODE additions:
 *   - effectiveFeeRate  : BNB-adjusted fee rate (GOD-3)
 *   - useBnbFeeDiscount : enable 0.075% fee with BNB (GOD-3)
 *   - discordWebhookUrl : fallback notification channel (GOD-6)
 */

export interface Candle {
  timestamp: number;
  open:      number;
  high:      number;
  low:       number;
  close:     number;
  volume:    number;
}

export enum Signal { HOLD = 0, BUY = 1, SELL = 2 }

export interface ModelOutput {
  signal:     Signal;
  probHold:   number;
  probBuy:    number;
  probSell:   number;
  confidence: number;
}

export interface Position {
  entryPrice:    number;
  entryTime:     number;
  size:          number;
  stopLoss:      number;
  takeProfit:    number;
  unrealizedPnl: number;
}

export interface TradeResult {
  side:       OrderSide;
  entryPrice: number;
  exitPrice:  number;
  size:       number;
  pnlPct:     number;
  pnlUsdt:    number;
  feePaid:    number;
  durationMs: number;
  exitReason: "TAKE_PROFIT" | "STOP_LOSS" | "SIGNAL" | "KILL_SWITCH";
}

export enum OrderSide   { BUY = "BUY", SELL = "SELL" }
export enum OrderStatus { PENDING = "PENDING", FILLED = "FILLED", CANCELED = "CANCELED", FAILED = "FAILED" }
export enum ExecutionMode { MOCK = "MOCK", PAPER = "PAPER", LIVE = "LIVE" }

export type FeatureVector = Float32Array;

function resolveMode(): ExecutionMode {
  if (process.env.MOCK_MODE   === "true") return ExecutionMode.MOCK;
  if (process.env.PAPER_TRADE === "true") return ExecutionMode.PAPER;
  return ExecutionMode.LIVE;
}

export const CONFIG = {
  // ── Exchange ──────────────────────────────────────────────────────────────
  apiKey:         process.env.BINANCE_API_KEY    ?? "",
  apiSecret:      process.env.BINANCE_API_SECRET ?? "",
  useTestnet:     process.env.USE_TESTNET === "true",

  // ── Trading ───────────────────────────────────────────────────────────────
  pair:           process.env.TRADING_PAIR ?? "BTC/USDT",
  timeframe:      process.env.TIMEFRAME    ?? "15m",
  windowSize:     parseInt(process.env.WINDOW_SIZE ?? "48", 10),

  // ── Risk ──────────────────────────────────────────────────────────────────
  stopLossPct:     parseFloat(process.env.STOP_LOSS_PCT      ?? "0.015"),
  takeProfitPct:   parseFloat(process.env.TAKE_PROFIT_PCT    ?? "0.03"),
  maxRiskPerTrade: parseFloat(process.env.MAX_RISK_PER_TRADE ?? "0.02"),
  minConfidence:   parseFloat(process.env.MIN_CONFIDENCE     ?? "0.60"),

  // ── GOD-3: Fee Configuration ──────────────────────────────────────────────
  // Set USE_BNB_FEE_DISCOUNT=true when your Binance account has BNB with
  // "Use BNB to pay fees" enabled → Binance charges 0.075% instead of 0.1%.
  // Must match your Python config.py USE_BNB_FEE_DISCOUNT setting.
  useBnbFeeDiscount: process.env.USE_BNB_FEE_DISCOUNT === "true",
  effectiveFeeRate:  parseFloat(
    process.env.EFFECTIVE_FEE_RATE ??
    (process.env.USE_BNB_FEE_DISCOUNT === "true" ? "0.00075" : "0.001")
  ),
  // MAKER-1: post-only limit orders. Rest on the passive side of the book so
  // every fill pays the MAKER fee with zero spread-crossing cost, at the price
  // of occasionally missing an entry when the market runs away (the fill loop
  // cancels on drift/timeout). On spot VIP0 maker==taker (this only saves the
  // spread); on USDT-M futures maker is 0.02% vs 0.05% taker — when enabling
  // there, set EFFECTIVE_FEE_RATE to match.
  useMakerOrders: process.env.USE_MAKER_ORDERS === "true",

  // ── Model ─────────────────────────────────────────────────────────────────
  modelPath:      process.env.MODEL_PATH ?? "",

  // ── Mode ──────────────────────────────────────────────────────────────────
  executionMode: resolveMode(),

  // ── Database ──────────────────────────────────────────────────────────────
  mongoUri:      process.env.MONGO_URI ?? "mongodb://localhost:27017/tradebot",

  // ── Notifications ─────────────────────────────────────────────────────────
  telegramToken:     process.env.TELEGRAM_BOT_TOKEN  ?? "",
  telegramChatId:    process.env.TELEGRAM_CHAT_ID    ?? "",
  // GOD-6: Discord webhook as fallback when Telegram is unavailable
  discordWebhookUrl: process.env.DISCORD_WEBHOOK_URL ?? "",
} as const;
