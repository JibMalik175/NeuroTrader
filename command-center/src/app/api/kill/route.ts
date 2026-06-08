/**
 * /api/kill  — Kill Switch API
 * ─────────────────────────────
 * Next.js App Router API route for the emergency kill switch.
 *
 * POST: Writes a kill signal document to MongoDB.
 *       The execution engine checks this collection periodically
 *       and halts all trading when a signal is found.
 *
 * GET:  Returns the current kill switch state.
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

// ── Kill Signal Schema ────────────────────────────────────────────────────────

const killSignalSchema = new mongoose.Schema(
  {
    active:    { type: Boolean, required: true, default: true },
    reason:    { type: String,  required: true },
    triggeredBy: { type: String, default: "dashboard" },
    triggeredAt: { type: Date,   default: Date.now },
    clearedAt:   { type: Date },
  },
  { timestamps: true },
);

const KillSignal =
  mongoose.models.KillSignal ?? mongoose.model("KillSignal", killSignalSchema);

// ── POST /api/kill ────────────────────────────────────────────────────────────

export async function POST(request: Request): Promise<NextResponse> {
  try {
    await ensureConnection();

    const body = await request.json().catch(() => ({}));
    const reason = body.reason ?? "Manual kill switch from dashboard";

    // Create kill signal document
    await KillSignal.create({
      active:      true,
      reason,
      triggeredBy: "dashboard",
      triggeredAt: new Date(),
    });

    console.log(`[API /kill] Kill switch ACTIVATED: ${reason}`);

    return NextResponse.json({
      success: true,
      message: "Kill switch activated — bot will halt trading",
      reason,
      timestamp: new Date().toISOString(),
    });

  } catch (err: any) {
    console.error("[API /kill] Error:", err.message);
    return NextResponse.json(
      { error: "Failed to activate kill switch", details: err.message },
      { status: 500 },
    );
  }
}

// ── GET /api/kill ─────────────────────────────────────────────────────────────

export async function GET(): Promise<NextResponse> {
  try {
    await ensureConnection();

    const latestSignal = await KillSignal.findOne({ active: true })
      .sort({ triggeredAt: -1 })
      .lean();

    return NextResponse.json({
      killed:    !!latestSignal,
      signal:    latestSignal ?? null,
      timestamp: new Date().toISOString(),
    });

  } catch (err: any) {
    console.error("[API /kill] Error:", err.message);
    return NextResponse.json(
      { error: "Failed to check kill switch state", details: err.message },
      { status: 500 },
    );
  }
}

// ── DELETE /api/kill  — Clear the kill switch ─────────────────────────────────

export async function DELETE(): Promise<NextResponse> {
  try {
    await ensureConnection();

    await KillSignal.updateMany(
      { active: true },
      { $set: { active: false, clearedAt: new Date() } },
    );

    console.log("[API /kill] Kill switch CLEARED");

    return NextResponse.json({
      success:   true,
      message:   "Kill switch cleared — bot may resume on next restart",
      timestamp: new Date().toISOString(),
    });

  } catch (err: any) {
    console.error("[API /kill] Error:", err.message);
    return NextResponse.json(
      { error: "Failed to clear kill switch", details: err.message },
      { status: 500 },
    );
  }
}
