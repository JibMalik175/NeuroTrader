/**
 * inference.ts  (The Strategist Brain)
 * ──────────────────────────────────────
 * Loads the ONNX model exported from Python and runs inference
 * on every new closed candle. Returns a typed ModelOutput with
 * the agent's decision and confidence level.
 *
 * The model was exported with input: "observation" [batch, obs_dim]
 * and output: "action_and_probs" [batch, 4]
 *   → [action_idx, p_hold, p_buy, p_sell]
 */

import * as ort from "onnxruntime-node";
import * as path from "path";
import * as fs from "fs";
import { Candle, Signal, ModelOutput, CONFIG } from "../utils/types";
import { buildObservationTensor } from "./indicators";
import { logger } from "../utils/logger";

// ── Action map ────────────────────────────────────────────────────────────────

const ACTION_NAMES: Record<number, string> = {
  0: "HOLD",
  1: "BUY",
  2: "SELL",
};

// ── Inference Engine ──────────────────────────────────────────────────────────

export class InferenceEngine {
  private session:    ort.InferenceSession | null = null;
  private obsShape:   number[]                    = [];
  private isLoaded:   boolean                     = false;

  async load(modelPath?: string): Promise<void> {
    const resolvedPath = modelPath
      || path.resolve(__dirname, "./models/tradebot.onnx");

    if (!fs.existsSync(resolvedPath)) {
      throw new Error(
        `ONNX model not found at: ${resolvedPath}\n` +
        `Run export_onnx.py in ai-training/ first, then copy the .onnx file here.`
      );
    }

    logger.info(`[Inference] Loading ONNX model: ${resolvedPath}`);

    this.session = await ort.InferenceSession.create(resolvedPath, {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all",
    });

    // Validate input shape from model metadata
    const inputMeta = this.session.inputNames;
    logger.info(`[Inference] Model inputs : ${inputMeta.join(", ")}`);
    logger.info(`[Inference] Model outputs: ${this.session.outputNames.join(", ")}`);

    this.isLoaded = true;
    logger.info("[Inference] Model loaded and ready");
  }

  /**
   * Runs a single inference pass.
   *
   * @param candles      - Full rolling window of candles from the Watcher
   * @param positionHeld - Is the bot currently in a trade?
   * @param entryPrice   - Entry price of current position (0 if flat)
   * @param peakBalance  - Highest balance seen in this session (for drawdown)
   * @param balance      - Current portfolio balance
   */
  async predict(
    candles:       Candle[],
    positionHeld:  boolean,
    entryPrice:    number,
    peakBalance:   number,
    balance:       number,
  ): Promise<ModelOutput> {
    if (!this.session || !this.isLoaded) {
      throw new Error("[Inference] Model not loaded. Call load() first.");
    }

    // Build observation tensor [1, obs_dim]
    const obsFlat = buildObservationTensor(
      candles,
      positionHeld,
      entryPrice,
      peakBalance,
      balance
    );
    const obsTensor = new ort.Tensor("float32", obsFlat, [1, obsFlat.length]);

    // Run inference
    const feeds   = { observation: obsTensor };
    const results = await this.session.run(feeds);
    const output  = results["action_and_probs"];

    if (!output || !output.data) {
      throw new Error("[Inference] Unexpected model output structure");
    }

    const data      = output.data as Float32Array;
    const actionIdx = Math.round(data[0]);
    const probHold  = data[1];
    const probBuy   = data[2];
    const probSell  = data[3];
    const confidence = Math.max(probHold, probBuy, probSell);

    const modelOutput: ModelOutput = {
      signal:     actionIdx as Signal,
      probHold,
      probBuy,
      probSell,
      confidence,
    };

    logger.debug("[Inference] Prediction", {
      action:     ACTION_NAMES[actionIdx],
      confidence: (confidence * 100).toFixed(1) + "%",
      probHold:   (probHold  * 100).toFixed(1) + "%",
      probBuy:    (probBuy   * 100).toFixed(1) + "%",
      probSell:   (probSell  * 100).toFixed(1) + "%",
    });

    return modelOutput;
  }

  get loaded(): boolean {
    return this.isLoaded;
  }
}
