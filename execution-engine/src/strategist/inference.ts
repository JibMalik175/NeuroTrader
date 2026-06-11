/**
 * inference.ts  (The Strategist Brain — recurrent contract)
 * ──────────────────────────────────────────────────────────
 * Runs the RecurrentPPO (LSTM) model exported by
 * ai-training/scripts/export_onnx_recurrent.py:
 *
 *   inputs : obs  [1, obs_dim]      raw features (VecNormalize is BAKED IN)
 *            h_in [layers, 1, size] LSTM hidden state
 *            c_in [layers, 1, size] LSTM cell state
 *   outputs: probs [1, 3]           (HOLD, BUY, SELL)
 *            h_out, c_out           updated state
 *
 * THE STATE IS THE MEMORY. The engine feeds h/c back on every candle —
 * zeros at process start, exactly like episode_start=True in training
 * eval. Dropping the state turns the LSTM into a goldfish (plain PPO
 * collapsed in the algo A/B: gross PF 0.058 vs 1.006).
 *
 * After a restart the memory is blank and rebuilds as candles stream in;
 * the model was trained with 48-candle windows, so it has its bearings
 * within a couple of days of 1h candles. The decision journal records a
 * "memory_reset" so month-end review can see restart boundaries.
 */

import * as ort from "onnxruntime-node";
import * as path from "path";
import * as fs from "fs";
import { Candle, Signal, ModelOutput, CONFIG } from "../utils/types";
import { buildObservationTensor } from "./indicators";
import { journal } from "../utils/decisionJournal";
import { logger } from "../utils/logger";

const ACTION_NAMES: Record<number, string> = { 0: "HOLD", 1: "BUY", 2: "SELL" };

// Must match the exporter's printout ([MODEL] obs_dim=1543 lstm=128x1).
// Overridable for future architectures without a code change.
const LSTM_SIZE   = parseInt(process.env.LSTM_SIZE   ?? "128", 10);
const LSTM_LAYERS = parseInt(process.env.LSTM_LAYERS ?? "1", 10);

export class InferenceEngine {
  private session:  ort.InferenceSession | null = null;
  private isLoaded: boolean = false;
  private h: ort.Tensor | null = null;
  private c: ort.Tensor | null = null;

  async load(modelPath?: string): Promise<void> {
    const resolvedPath = modelPath
      || path.resolve(__dirname, "./models/tradebot.onnx");

    if (!fs.existsSync(resolvedPath)) {
      throw new Error(
        `ONNX model not found at: ${resolvedPath}\n` +
        `Run ai-training/scripts/export_onnx_recurrent.py first.`
      );
    }

    logger.info(`[Inference] Loading ONNX model: ${resolvedPath}`);
    this.session = await ort.InferenceSession.create(resolvedPath, {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all",
    });

    logger.info(`[Inference] Model inputs : ${this.session.inputNames.join(", ")}`);
    logger.info(`[Inference] Model outputs: ${this.session.outputNames.join(", ")}`);

    // Refuse the legacy stateless export — running the LSTM without its
    // state would silently produce garbage decisions, not an error.
    if (!this.session.inputNames.includes("h_in")) {
      throw new Error(
        "[Inference] This model uses the LEGACY stateless contract. The engine " +
        "requires the recurrent export (obs/h_in/c_in → probs/h_out/c_out). " +
        "Re-export with export_onnx_recurrent.py."
      );
    }

    this.resetMemory("model load");
    this.isLoaded = true;
    logger.info(`[Inference] Recurrent model ready (lstm ${LSTM_SIZE}x${LSTM_LAYERS})`);
  }

  /** Zero the LSTM state (process start / explicit reset). */
  resetMemory(reason: string): void {
    const zeros = () => new ort.Tensor(
      "float32",
      new Float32Array(LSTM_LAYERS * 1 * LSTM_SIZE),
      [LSTM_LAYERS, 1, LSTM_SIZE],
    );
    this.h = zeros();
    this.c = zeros();
    journal.log("memory_reset", { reason });
    logger.info(`[Inference] LSTM memory reset (${reason})`);
  }

  async predict(
    candles:       Candle[],
    positionHeld:  boolean,
    entryPrice:    number,
    peakBalance:   number,
    balance:       number,
  ): Promise<ModelOutput> {
    if (!this.session || !this.isLoaded || !this.h || !this.c) {
      throw new Error("[Inference] Model not loaded. Call load() first.");
    }

    // Raw features — the export normalizes internally (VecNormalize baked in)
    const obsFlat = buildObservationTensor(
      candles, positionHeld, entryPrice, peakBalance, balance
    );
    const obsTensor = new ort.Tensor("float32", obsFlat, [1, obsFlat.length]);

    const results = await this.session.run({
      obs: obsTensor, h_in: this.h, c_in: this.c,
    });

    const probsOut = results["probs"];
    if (!probsOut?.data || !results["h_out"] || !results["c_out"]) {
      throw new Error("[Inference] Unexpected model output structure");
    }

    // Carry the memory to the next candle
    this.h = results["h_out"] as ort.Tensor;
    this.c = results["c_out"] as ort.Tensor;

    const probs = probsOut.data as Float32Array;
    const probHold = probs[0], probBuy = probs[1], probSell = probs[2];
    const confidence = Math.max(probHold, probBuy, probSell);
    const actionIdx =
      confidence === probHold ? 0 : confidence === probBuy ? 1 : 2;

    logger.debug("[Inference] Prediction", {
      action:     ACTION_NAMES[actionIdx],
      confidence: (confidence * 100).toFixed(1) + "%",
      probHold:   (probHold * 100).toFixed(1) + "%",
      probBuy:    (probBuy  * 100).toFixed(1) + "%",
      probSell:   (probSell * 100).toFixed(1) + "%",
    });

    return { signal: actionIdx as Signal, probHold, probBuy, probSell, confidence };
  }

  get loaded(): boolean {
    return this.isLoaded;
  }
}
