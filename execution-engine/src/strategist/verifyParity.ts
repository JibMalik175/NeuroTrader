/**
 * verifyParity.ts  (FIX #3)
 * ─────────────────────────
 * Compares TypeScript indicator output against Python ground truth.
 *
 * Run AFTER generating the reference file:
 *   python ai-training/scripts/verify_feature_parity.py --input data/BTC_USDT_1h.parquet
 *
 * Then from execution-engine/:
 *   npx ts-node src/strategist/verifyParity.ts
 *
 * A PASS means the live bot feeds the ONNX model identical data to training.
 * A FAIL means concept drift — retrace the mismatched indicator formula.
 */

import * as fs   from "fs";
import * as path from "path";
import { buildObservationTensor } from "./indicators";
import { Candle } from "../utils/types";

const PARITY_FILE = path.resolve(__dirname, "../../../ai-training/data/parity_test.json");
const TOLERANCE   = 0.001;   // Max allowed absolute difference per feature

// Feature names in the SAME ORDER they appear in the flat Float32Array
// This must match the feature push order in indicators.ts buildObservationTensor()
const FEATURE_NAMES = [
  "log_return", "log_return_h", "log_return_l", "log_return_v",
  "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "candle_direction",
  "rsi",
  "macd", "macd_signal", "macd_hist",
  "bb_position", "bb_width",
  "atr_ratio",
  "ema_cross_short", "ema_cross_long",
  "volume_ratio",
];

interface ParityData {
  candles:           Candle[];
  expected_features: Record<string, number>;
  tolerance:         number;
  candle_count:      number;
}

function extractLastCandleFeatures(tensor: Float32Array, windowSize: number): Record<string, number> {
  const featuresPerCandle = FEATURE_NAMES.length;
  // The last candle's features occupy the final featuresPerCandle slots
  // before the 3 portfolio state values appended at the end
  const lastCandleStart = (windowSize - 1) * featuresPerCandle;
  const result: Record<string, number> = {};

  for (let i = 0; i < featuresPerCandle; i++) {
    result[FEATURE_NAMES[i]] = tensor[lastCandleStart + i];
  }
  return result;
}

async function runParityCheck(): Promise<void> {
  if (!fs.existsSync(PARITY_FILE)) {
    console.error(`\n[PARITY] Reference file not found: ${PARITY_FILE}`);
    console.error("  Run this first:");
    console.error("    cd ai-training");
    console.error("    python scripts/verify_feature_parity.py --input data/BTC_USDT_1h.parquet\n");
    process.exit(1);
  }

  const data: ParityData = JSON.parse(fs.readFileSync(PARITY_FILE, "utf-8"));
  const candles   = data.candles;
  const expected  = data.expected_features;
  const tolerance = data.tolerance ?? TOLERANCE;

  console.log(`\n${"=".repeat(60)}`);
  console.log("  FEATURE PARITY CHECK — Python vs TypeScript");
  console.log(`  Candles: ${candles.length} | Tolerance: ±${tolerance}`);
  console.log("=".repeat(60));

  // Build the full observation tensor using TypeScript indicators.ts
  // Use dummy portfolio state (flat, no position)
  const tensor = buildObservationTensor(
    candles,
    false,   // positionHeld
    0,       // entryPrice
    10_000,  // peakBalance
    10_000,  // currentBalance
  );

  const windowSize = candles.length;   // parity test uses all candles as window
  const actual = extractLastCandleFeatures(tensor, windowSize);

  let passed = 0;
  let failed = 0;
  const failures: string[] = [];

  console.log(`\n  ${"Feature".padEnd(22)} ${"Python".padStart(14)} ${"TypeScript".padStart(14)} ${"Delta".padStart(12)} ${"Status".padStart(8)}`);
  console.log("  " + "-".repeat(72));

  for (const name of FEATURE_NAMES) {
    const exp  = expected[name] ?? 0;
    const act  = actual[name]   ?? 0;
    const diff = Math.abs(exp - act);
    const ok   = diff <= tolerance || (Math.abs(exp) < 1e-10 && Math.abs(act) < 1e-10);

    const status = ok ? "✅ PASS" : "❌ FAIL";
    const delta  = diff.toFixed(8);

    console.log(`  ${name.padEnd(22)} ${exp.toFixed(8).padStart(14)} ${act.toFixed(8).padStart(14)} ${delta.padStart(12)} ${status.padStart(8)}`);

    if (ok) { passed++; }
    else    { failed++; failures.push(`${name}: Δ=${delta} (py=${exp.toFixed(8)}, ts=${act.toFixed(8)})`); }
  }

  console.log("\n" + "=".repeat(60));

  if (failed === 0) {
    console.log(`  ✅ ALL ${passed} FEATURES PASS — TypeScript ≡ Python`);
    console.log("     Safe to deploy. Model will see identical inputs in production.");
  } else {
    console.log(`  ❌ ${failed} FEATURE(S) FAILED — CONCEPT DRIFT DETECTED`);
    console.log("     The live bot will feed different data to the ONNX model than it was trained on.");
    console.log("     Fix the mismatched formula(s) in indicators.ts before going live.\n");
    console.log("  Failed features:");
    failures.forEach(f => console.log(`    • ${f}`));
    console.log("");
    process.exit(1);
  }

  console.log("=".repeat(60) + "\n");
}

runParityCheck().catch(err => {
  console.error("[PARITY] Unexpected error:", err.message);
  process.exit(1);
});
