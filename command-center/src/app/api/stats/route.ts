/**
 * /api/stats  — Trading Statistics API
 * ─────────────────────────────────────
 * Next.js App Router API route that queries MongoDB for
 * all trading metrics displayed on the command-center dashboard.
 *
 * Returns:
 *   balance, totalPnl, winRate, totalTrades, profitFactor,
 *   avgWin, avgLoss, trades[], equityCurve[]
 */

import { NextResponse } from "next/server";
import mongoose from "mongoose";

// ── MongoDB Connection ────────────────────────────────────────────────────────

const MONGO_URI = process.env.MONGO_URI ?? "mongodb://localhost:27017/tradebot";

let isConnected = false;

async function ensureConnection(): Promise<void> {
  if (isConnected && mongoose.connection.readyState >= 1) return;

  await mongoose.connect(MONGO_URI, {
    serverSelectionTimeoutMS: 5000,
  });
  isConnected = true;
}

// ── Schemas (lightweight redefinition for the command-center) ──────────────────

const tradeSchema = new mongoose.Schema(
  {
    pair:        String,
    mode:        String,
    side:        String,
    entryPrice:  Number,
    exitPrice:   Number,
    size:        Number,
    pnlPct:      Number,
    pnlUsdt:     Number,
    feePaid:     Number,
    stopLoss:    Number,
    takeProfit:  Number,
    entryTime:   Date,
    exitTime:    Date,
    durationMs:  Number,
    exitReason:  String,
    orderId:     String,
  },
  { timestamps: true },
);

const snapshotSchema = new mongoose.Schema(
  {
    timestamp:     Date,
    mode:          String,
    totalBalance:  Number,
    availableUsdt: Number,
    positionValue: Number,
    totalPnlPct:   Number,
    openPosition:  Boolean,
  },
  { timestamps: true },
);

const Trade    = mongoose.models.Trade    ?? mongoose.model("Trade", tradeSchema);
const Snapshot = mongoose.models.Snapshot ?? mongoose.model("Snapshot", snapshotSchema);

// ── GET /api/stats ────────────────────────────────────────────────────────────

export async function GET(): Promise<NextResponse> {
  try {
    await ensureConnection();

    // Fetch all trades, most recent first
    const trades = await Trade.find({}).sort({ exitTime: -1 }).lean();

    const totalTrades = trades.length;

    // Shakedown fix: the page expects { stats, equityCurve[{time,balance}],
    // recentTrades } with display-ready strings — the old flat/numeric shape
    // rendered as permanent dashes. Shape everything here, once.
    if (totalTrades === 0) {
      const latest = (await Snapshot.findOne({}).sort({ timestamp: -1 }).lean()) as any;
      return NextResponse.json({
        stats: {
          totalTrades:  "0",
          winRate:      "0.0",
          totalPnlUsdt: "0.00",
          profitFactor: "0.00",
          avgWinPct:    "0.00",
          avgLossPct:   "0.00",
          balance:      (latest?.totalBalance ?? 10_000).toFixed(2),
          inPosition:   latest?.openPosition ?? false,
        },
        equityCurve:  [],
        recentTrades: [],
      });
    }

    // ── Compute stats ───────────────────────────────────────────────────────

    const wins   = trades.filter((t: any) => t.pnlUsdt > 0);
    const losses = trades.filter((t: any) => t.pnlUsdt <= 0);

    const totalPnl   = trades.reduce((sum: number, t: any) => sum + (t.pnlUsdt ?? 0), 0);
    const winRate     = totalTrades > 0 ? wins.length / totalTrades : 0;

    const grossProfit = wins.reduce((sum: number, t: any) => sum + t.pnlUsdt, 0);
    const grossLoss   = Math.abs(losses.reduce((sum: number, t: any) => sum + t.pnlUsdt, 0));

    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;

    // ── Equity curve from snapshots ─────────────────────────────────────────

    const snapshots = await Snapshot.find({})
      .sort({ timestamp: 1 })
      .select({ timestamp: 1, totalBalance: 1, totalPnlPct: 1 })
      .lean();

    const equityCurve = snapshots.map((s: any) => ({
      time:    new Date(s.timestamp).toLocaleString("en-GB", {
                 month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" }),
      balance: s.totalBalance,
    }));

    // Latest balance from the most recent snapshot, or compute from trades
    const latestSnapshot = snapshots[snapshots.length - 1] as any;
    const balance = latestSnapshot?.totalBalance ?? 10_000 + totalPnl;

    // avgWin/avgLoss as PERCENT of trade (page label says %), not USDT
    const avgWinPct  = wins.length   ? wins.reduce((s: number, t: any) => s + (t.pnlPct ?? 0), 0)   / wins.length   * 100 : 0;
    const avgLossPct = losses.length ? losses.reduce((s: number, t: any) => s + (t.pnlPct ?? 0), 0) / losses.length * 100 : 0;

    return NextResponse.json({
      stats: {
        totalTrades:  String(totalTrades),
        winRate:      (winRate * 100).toFixed(1),
        totalPnlUsdt: totalPnl.toFixed(2),
        profitFactor: Number.isFinite(profitFactor) ? profitFactor.toFixed(2) : "∞",
        avgWinPct:    avgWinPct.toFixed(2),
        avgLossPct:   avgLossPct.toFixed(2),
        balance:      balance.toFixed(2),
        inPosition:   latestSnapshot?.openPosition ?? false,
      },
      equityCurve,
      recentTrades: trades.slice(0, 100).map((t: any) => ({
        id:         String(t._id),
        pair:       t.pair,
        side:       t.side ?? "buy",
        entryPrice: t.entryPrice,
        exitPrice:  t.exitPrice,
        pnlPct:     ((t.pnlPct ?? 0) * 100).toFixed(3),
        pnlUsdt:    (t.pnlUsdt ?? 0).toFixed(2),
        exitReason: t.exitReason ?? "?",
        exitTime:   t.exitTime ? new Date(t.exitTime).toLocaleString("en-GB", {
                      month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" }) : "—",
      })),
    });

  } catch (err: any) {
    console.error("[API /stats] Error:", err.message);
    return NextResponse.json(
      { error: "Failed to fetch trading stats", details: err.message },
      { status: 500 },
    );
  }
}
