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

    if (totalTrades === 0) {
      return NextResponse.json({
        balance:      0,
        totalPnl:     0,
        winRate:      0,
        totalTrades:  0,
        profitFactor: 0,
        avgWin:       0,
        avgLoss:      0,
        trades:       [],
        equityCurve:  [],
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
    const avgWin       = wins.length   > 0 ? grossProfit / wins.length          : 0;
    const avgLoss      = losses.length > 0 ? grossLoss   / losses.length        : 0;

    // ── Equity curve from snapshots ─────────────────────────────────────────

    const snapshots = await Snapshot.find({})
      .sort({ timestamp: 1 })
      .select({ timestamp: 1, totalBalance: 1, totalPnlPct: 1 })
      .lean();

    const equityCurve = snapshots.map((s: any) => ({
      timestamp: s.timestamp,
      balance:   s.totalBalance,
      pnlPct:    s.totalPnlPct,
    }));

    // Latest balance from the most recent snapshot, or compute from trades
    const latestSnapshot = snapshots[snapshots.length - 1] as any;
    const balance = latestSnapshot?.totalBalance ?? 10_000 + totalPnl;

    return NextResponse.json({
      balance:      Math.round(balance * 100) / 100,
      totalPnl:     Math.round(totalPnl * 100) / 100,
      winRate:      Math.round(winRate * 1000) / 1000,
      totalTrades,
      profitFactor: Math.round(profitFactor * 100) / 100,
      avgWin:       Math.round(avgWin * 100) / 100,
      avgLoss:      Math.round(avgLoss * 100) / 100,
      trades:       trades.slice(0, 100),   // Cap at 100 for performance
      equityCurve,
    });

  } catch (err: any) {
    console.error("[API /stats] Error:", err.message);
    return NextResponse.json(
      { error: "Failed to fetch trading stats", details: err.message },
      { status: 500 },
    );
  }
}
