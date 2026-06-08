/**
 * logger.ts
 * ─────────
 * Winston-based logger used across the entire execution engine.
 * Logs to both console (colorized) and a rotating file for
 * post-mortem analysis and debugging.
 *
 * Usage:
 *   import { logger } from "../utils/logger";
 *   logger.info("message", { key: value });
 */

import winston from "winston";
import path from "path";
import fs from "fs";

// ── Ensure log directory exists ──────────────────────────────────────────────

const LOG_DIR  = path.resolve(__dirname, "../../logs");
const LOG_FILE = path.join(LOG_DIR, "tradebot.log");

if (!fs.existsSync(LOG_DIR)) {
  fs.mkdirSync(LOG_DIR, { recursive: true });
}

// ── Custom format ────────────────────────────────────────────────────────────

const logFormat = winston.format.combine(
  winston.format.timestamp({ format: "YYYY-MM-DD HH:mm:ss.SSS" }),
  winston.format.errors({ stack: true }),
  winston.format.printf(({ timestamp, level, message, ...meta }) => {
    const metaStr = Object.keys(meta).length ? " " + JSON.stringify(meta) : "";
    return `${timestamp} [${level.toUpperCase().padEnd(5)}] ${message}${metaStr}`;
  }),
);

// ── Logger instance ──────────────────────────────────────────────────────────

export const logger = winston.createLogger({
  level: process.env.LOG_LEVEL ?? "info",
  format: logFormat,
  transports: [
    // Console — colorized for readability
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize({ all: true }),
        logFormat,
      ),
    }),

    // File — plain text for log aggregation / grep
    new winston.transports.File({
      filename:  LOG_FILE,
      maxsize:   10 * 1024 * 1024,   // 10 MB per file
      maxFiles:  5,                   // Keep last 5 rotated files
      tailable:  true,
    }),
  ],

  // Never crash on a logging error
  exitOnError: false,
});

// Log startup
logger.debug(`[Logger] Initialized — file: ${LOG_FILE}`);
