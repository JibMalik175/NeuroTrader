/**
 * notifier.ts  (GOD-6)
 * ──────────────────────
 * Multi-channel alerts: Telegram primary, Discord fallback.
 *
 * GOD-6: Previously Telegram-only. If Telegram is down (it happens),
 * you get zero alerts — you won't know if your bot took a loss.
 * Discord webhook added as automatic fallback. Both channels receive
 * all alerts simultaneously when both are configured.
 *
 * New alert: sendCircuitBreakerAlert — fires when daily loss limit hit.
 */

import { logger } from "./logger";
import { CONFIG, ExecutionMode } from "./types";

interface BuyAlert  { pair: string; price: number; size: number; stopLoss: number; takeProfit: number; confidence: number; mode: ExecutionMode; }
interface SellAlert { pair: string; price: number; pnlPct: number; pnlUsdt: number; exitReason: string; mode: ExecutionMode; newBalance: number; }

export class Notifier {
  private telegramEnabled: boolean;
  private discordEnabled:  boolean;
  private telegramUrl:     string;
  private chatId:          string;
  private discordUrl:      string;

  constructor() {
    this.telegramEnabled = !!(CONFIG.telegramToken && CONFIG.telegramChatId);
    this.discordEnabled  = !!CONFIG.discordWebhookUrl;
    this.telegramUrl     = `https://api.telegram.org/bot${CONFIG.telegramToken}`;
    this.chatId          = CONFIG.telegramChatId;
    this.discordUrl      = CONFIG.discordWebhookUrl;

    const channels = [
      this.telegramEnabled ? "Telegram ✅" : "Telegram ❌",
      this.discordEnabled  ? "Discord ✅"  : "Discord ❌",
    ].join(" | ");
    logger.info(`[Notifier] ${channels}`);

    if (!this.telegramEnabled && !this.discordEnabled) {
      logger.warn("[Notifier] No notification channels configured — alerts suppressed");
    }
  }

  // ── Trade Alerts ──────────────────────────────────────────────────────────

  async sendTradeAlert(side: "BUY" | "SELL", data: BuyAlert | SellAlert): Promise<void> {
    let tgMsg: string;
    let discordMsg: string;

    if (side === "BUY") {
      const d = data as BuyAlert;
      tgMsg = [
        `🟢 *TRADE ENTERED*`, ``,
        `📊 Pair: \`${d.pair}\``,
        `💰 Entry: \`$${d.price.toFixed(4)}\``,
        `📦 Size: \`${d.size.toFixed(6)}\``,
        `🛑 Stop-Loss: \`$${d.stopLoss.toFixed(4)}\``,
        `🎯 Take-Profit: \`$${d.takeProfit.toFixed(4)}\``,
        `🤖 Confidence: \`${(d.confidence * 100).toFixed(1)}%\``,
        `⚙️ Mode: \`${d.mode}\``,
      ].join("\n");
      discordMsg = `🟢 **TRADE ENTERED** | ${d.pair} | Entry: $${d.price.toFixed(4)} | SL: $${d.stopLoss.toFixed(4)} | TP: $${d.takeProfit.toFixed(4)} | Conf: ${(d.confidence*100).toFixed(1)}%`;
    } else {
      const d = data as SellAlert;
      const emoji = d.pnlUsdt >= 0 ? "✅" : "❌";
      const sign  = d.pnlUsdt >= 0 ? "+" : "";
      tgMsg = [
        `${emoji} *TRADE CLOSED*`, ``,
        `📊 Pair: \`${d.pair}\``,
        `💰 Exit: \`$${d.price.toFixed(4)}\``,
        `📈 PnL: \`${sign}${(d.pnlPct * 100).toFixed(3)}% (${sign}$${d.pnlUsdt.toFixed(4)})\``,
        `📝 Reason: \`${d.exitReason}\``,
        `💼 Balance: \`$${d.newBalance.toFixed(2)}\``,
        `⚙️ Mode: \`${d.mode}\``,
      ].join("\n");
      discordMsg = `${emoji} **TRADE CLOSED** | ${d.pair} | PnL: ${sign}${(d.pnlPct*100).toFixed(3)}% (${sign}$${d.pnlUsdt.toFixed(4)}) | Reason: ${d.exitReason} | Balance: $${d.newBalance.toFixed(2)}`;
    }

    await this.sendAll(tgMsg, discordMsg);
  }

  async sendAlert(message: string): Promise<void> {
    await this.sendAll(message, message.replace(/[*`]/g, "**"));
  }

  async sendCircuitBreakerAlert(drawdownPct: number, balance: number): Promise<void> {
    const tgMsg = [
      `⚠️ *CIRCUIT BREAKER TRIPPED*`, ``,
      `Daily loss: \`${(drawdownPct * 100).toFixed(2)}%\` exceeded 5% limit`,
      `Balance: \`$${balance.toFixed(2)}\``,
      `Trading halted until UTC midnight.`,
    ].join("\n");
    const dcMsg = `⚠️ **CIRCUIT BREAKER** | Daily loss ${(drawdownPct*100).toFixed(2)}% | Balance $${balance.toFixed(2)} | Halted until midnight`;
    await this.sendAll(tgMsg, dcMsg);
  }

  async sendDailySummary(stats: { totalTrades: number; winRate: number; totalPnl: number; balance: number; }): Promise<void> {
    const sign  = stats.totalPnl >= 0 ? "+" : "";
    const emoji = stats.totalPnl >= 0 ? "📈" : "📉";
    const tgMsg = [
      `${emoji} *DAILY SUMMARY*`, ``,
      `🔄 Trades: \`${stats.totalTrades}\``,
      `🎯 Win Rate: \`${(stats.winRate * 100).toFixed(1)}%\``,
      `💰 PnL: \`${sign}$${stats.totalPnl.toFixed(2)}\``,
      `💼 Balance: \`$${stats.balance.toFixed(2)}\``,
    ].join("\n");
    const dcMsg = `${emoji} **DAILY** | Trades: ${stats.totalTrades} | WR: ${(stats.winRate*100).toFixed(1)}% | PnL: ${sign}$${stats.totalPnl.toFixed(2)} | Bal: $${stats.balance.toFixed(2)}`;
    await this.sendAll(tgMsg, dcMsg);
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  private async sendAll(telegramText: string, discordText: string): Promise<void> {
    const tasks: Promise<void>[] = [];
    if (this.telegramEnabled) tasks.push(this.sendTelegram(telegramText));
    if (this.discordEnabled)  tasks.push(this.sendDiscord(discordText));
    await Promise.allSettled(tasks);
  }

  private async sendTelegram(text: string): Promise<void> {
    try {
      const res = await fetch(`${this.telegramUrl}/sendMessage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chat_id: this.chatId, text, parse_mode: "Markdown" }),
      });
      if (!res.ok) logger.warn(`[Notifier] Telegram failed: ${await res.text()}`);
    } catch (err: any) {
      logger.warn(`[Notifier] Telegram error: ${err.message}`);
    }
  }

  private async sendDiscord(content: string): Promise<void> {
    try {
      const res = await fetch(this.discordUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });
      if (!res.ok) logger.warn(`[Notifier] Discord failed: ${await res.text()}`);
    } catch (err: any) {
      logger.warn(`[Notifier] Discord error: ${err.message}`);
    }
  }
}
