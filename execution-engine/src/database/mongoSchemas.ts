/**
 * mongoSchemas.ts
 * ───────────────
 * Mongoose schemas and models for persisting trading data.
 * Also exports connectDB / disconnectDB for lifecycle management.
 *
 * Collections:
 *   trades      — Completed trade records (entry + exit + PnL)
 *   snapshots   — Portfolio snapshots after each trade
 *   signallogs  — Every AI signal (for backtesting / auditing)
 *   ticks       — Closed candle OHLCV data (for the dashboard)
 */

import mongoose, { Schema, Document, Model } from "mongoose";
import { CONFIG } from "../utils/types";
import { logger } from "../utils/logger";

// ── Connection Helpers ────────────────────────────────────────────────────────

export async function connectDB(): Promise<void> {
  if (mongoose.connection.readyState >= 1) return;

  await mongoose.connect(CONFIG.mongoUri, {
    // Modern Mongoose 7+ uses these defaults; explicit for clarity
    serverSelectionTimeoutMS: 5000,
    socketTimeoutMS:          45000,
  });

  logger.info(`[DB] Connected to MongoDB: ${CONFIG.mongoUri}`);
}

export async function disconnectDB(): Promise<void> {
  if (mongoose.connection.readyState === 0) return;
  await mongoose.disconnect();
  logger.info("[DB] MongoDB disconnected");
}

// ── Trade Schema ──────────────────────────────────────────────────────────────
// Used by: executioner.ts → persistTrade()

export interface ITrade extends Document {
  pair:        string;
  mode:        string;
  side:        string;
  entryPrice:  number;
  exitPrice:   number;
  size:        number;
  pnlPct:      number;
  pnlUsdt:     number;
  feePaid:     number;
  stopLoss:    number;
  takeProfit:  number;
  entryTime:   Date;
  exitTime:    Date;
  durationMs:  number;
  exitReason:  string;
  orderId?:    string;
}

const tradeSchema = new Schema<ITrade>(
  {
    pair:        { type: String, required: true, index: true },
    mode:        { type: String, required: true },
    side:        { type: String, required: true },
    entryPrice:  { type: Number, required: true },
    exitPrice:   { type: Number, required: true },
    size:        { type: Number, required: true },
    pnlPct:      { type: Number, required: true },
    pnlUsdt:     { type: Number, required: true },
    feePaid:     { type: Number, default: 0 },
    stopLoss:    { type: Number, default: 0 },
    takeProfit:  { type: Number, default: 0 },
    entryTime:   { type: Date,   required: true },
    exitTime:    { type: Date,   required: true },
    durationMs:  { type: Number, required: true },
    exitReason:  { type: String, required: true },
    orderId:     { type: String },
  },
  { timestamps: true },
);

// ── Snapshot Schema ───────────────────────────────────────────────────────────
// Used by: executioner.ts → saveSnapshot()

export interface ISnapshot extends Document {
  timestamp:     Date;
  mode:          string;
  totalBalance:  number;
  availableUsdt: number;
  positionValue: number;
  totalPnlPct:   number;
  openPosition:  boolean;
}

const snapshotSchema = new Schema<ISnapshot>(
  {
    timestamp:     { type: Date,    required: true, index: true },
    mode:          { type: String,  required: true },
    totalBalance:  { type: Number,  required: true },
    availableUsdt: { type: Number,  required: true },
    positionValue: { type: Number,  default: 0 },
    totalPnlPct:   { type: Number,  default: 0 },
    openPosition:  { type: Boolean, default: false },
  },
  { timestamps: true },
);

// ── SignalLog Schema ──────────────────────────────────────────────────────────
// Used by: executioner.ts → persistSignal(), updateSignalSkipReason()

export interface ISignalLog extends Document {
  pair:        string;
  timestamp:   Date;
  signal:      number;
  probHold:    number;
  probBuy:     number;
  probSell:    number;
  confidence:  number;
  acted:       boolean;
  skipReason?: string;
}

const signalLogSchema = new Schema<ISignalLog>(
  {
    pair:        { type: String, required: true, index: true },
    timestamp:   { type: Date,   required: true, index: true },
    signal:      { type: Number, required: true },
    probHold:    { type: Number, required: true },
    probBuy:     { type: Number, required: true },
    probSell:    { type: Number, required: true },
    confidence:  { type: Number, required: true },
    acted:       { type: Boolean, default: false },
    skipReason:  { type: String },
  },
  { timestamps: true },
);

// ── Tick Schema ───────────────────────────────────────────────────────────────
// Used by: binanceStream.ts → persistCandle()

export interface ITick extends Document {
  pair:      string;
  timeframe: string;
  timestamp: Date;
  open:      number;
  high:      number;
  low:       number;
  close:     number;
  volume:    number;
}

const tickSchema = new Schema<ITick>(
  {
    pair:      { type: String, required: true },
    timeframe: { type: String, required: true },
    timestamp: { type: Date,   required: true },
    open:      { type: Number, required: true },
    high:      { type: Number, required: true },
    low:       { type: Number, required: true },
    close:     { type: Number, required: true },
    volume:    { type: Number, required: true },
  },
  { timestamps: true },
);

// Compound index for upsert queries in binanceStream.ts
tickSchema.index({ pair: 1, timeframe: 1, timestamp: 1 }, { unique: true });

// ── KillSignal Schema ─────────────────────────────────────────────────────────
// Written by: command-center /api/kill (the dashboard's red button).
// Read by: index.ts kill-switch poll. Before this existed the dashboard
// button wrote documents NOTHING read — a placebo. Found in the pre-paper
// shakedown.

export interface IKillSignal extends Document {
  active:      boolean;
  reason:      string;
  triggeredBy: string;
  triggeredAt: Date;
  clearedAt?:  Date;
}

const killSignalSchema = new Schema<IKillSignal>(
  {
    active:      { type: Boolean, required: true, default: true, index: true },
    reason:      { type: String,  required: true },
    triggeredBy: { type: String,  default: "dashboard" },
    triggeredAt: { type: Date,    default: Date.now },
    clearedAt:   { type: Date },
  },
  { timestamps: true },
);

// ── Models ────────────────────────────────────────────────────────────────────

export const KillSignal: Model<IKillSignal> = mongoose.models.KillSignal ?? mongoose.model<IKillSignal>("KillSignal", killSignalSchema);
export const Trade:     Model<ITrade>     = mongoose.models.Trade     ?? mongoose.model<ITrade>("Trade", tradeSchema);
export const Snapshot:  Model<ISnapshot>  = mongoose.models.Snapshot  ?? mongoose.model<ISnapshot>("Snapshot", snapshotSchema);
export const SignalLog: Model<ISignalLog> = mongoose.models.SignalLog ?? mongoose.model<ISignalLog>("SignalLog", signalLogSchema);
export const Tick:      Model<ITick>      = mongoose.models.Tick      ?? mongoose.model<ITick>("Tick", tickSchema);
