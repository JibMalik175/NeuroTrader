/**
 * decisionJournal.ts — the paper-trading audit trail (month-1 verdict data)
 * ─────────────────────────────────────────────────────────────────────────
 * One JSON line per decision, appended to logs/journal/YYYY-MM-DD.jsonl
 * (UTC). File-based on purpose: MongoDB persistence exists but fails
 * SILENTLY when Mongo is down — this journal has no dependencies, survives
 * crashes (append-only), and is human-greppable.
 *
 * Entry types written by the executioner:
 *   signal        — every model output (probs, confidence) + context
 *   entry         — executed entry (actual fill price/size, SL/TP)
 *   entry_blocked — protections refused (reason)
 *   entry_skipped — low confidence / zero size
 *   exit          — closed trade (pnl, fees, reason, new balance)
 *   kill          — emergency stop
 *
 * Analyzed by ai-training/scripts/journal_report.py at month end.
 */

import fs   from "fs";
import path from "path";
import { logger } from "./logger";

export class DecisionJournal {
  private dir: string;
  private failed = false;

  constructor(dir = path.join(process.cwd(), "logs", "journal")) {
    this.dir = dir;
    try {
      fs.mkdirSync(this.dir, { recursive: true });
    } catch (e: any) {
      this.failed = true;
      logger.error(`[Journal] cannot create ${this.dir}: ${e.message} — journaling DISABLED`);
    }
  }

  /** Append one decision line. Never throws — trading must not die for a log. */
  log(type: string, data: Record<string, unknown>): void {
    if (this.failed) return;
    const day  = new Date().toISOString().slice(0, 10);
    const line = JSON.stringify({ ts: new Date().toISOString(), type, ...data });
    try {
      fs.appendFileSync(path.join(this.dir, `${day}.jsonl`), line + "\n");
    } catch (e: any) {
      // warn once per failure burst, keep trading
      logger.warn(`[Journal] append failed: ${e.message}`);
    }
  }
}

export const journal = new DecisionJournal();
