"use client";

import { useEffect, useState, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from "recharts";

// ── Types ─────────────────────────────────────────────────────────────────────

interface Stats {
  totalTrades:   string;
  winRate:       string;
  totalPnlUsdt:  string;
  profitFactor:  string;
  avgWinPct:     string;
  avgLossPct:    string;
  balance:       string;
  inPosition:    boolean;
}

interface EquityPoint  { time: string; balance: number; }
interface Trade {
  id:         string;
  pair:       string;
  side:       string;
  entryPrice: number;
  exitPrice:  number;
  pnlPct:     string;
  pnlUsdt:    string;
  exitReason: string;
  exitTime:   string;
}

// ── Stat Card ─────────────────────────────────────────────────────────────────

function StatCard({
  label, value, sub, color,
}: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div style={{
      background: "#1a1f2e", border: "1px solid #2a2f42",
      borderRadius: 12, padding: "20px 24px", minWidth: 160,
    }}>
      <p style={{ margin: 0, color: "#6b7280", fontSize: 12, textTransform: "uppercase", letterSpacing: 1 }}>
        {label}
      </p>
      <p style={{ margin: "8px 0 4px", fontSize: 26, fontWeight: 700, color: color || "#f1f5f9" }}>
        {value}
      </p>
      {sub && <p style={{ margin: 0, fontSize: 12, color: "#9ca3af" }}>{sub}</p>}
    </div>
  );
}

// ── Kill Switch Button ────────────────────────────────────────────────────────

function KillSwitch() {
  const [status, setStatus] = useState<"idle" | "confirm" | "sent">("idle");

  const handleClick = async () => {
    if (status === "idle") { setStatus("confirm"); return; }
    if (status === "confirm") {
      await fetch("/api/kill", { method: "POST" });
      setStatus("sent");
    }
  };

  const colors = {
    idle:    { bg: "#7f1d1d", border: "#ef4444", text: "#fca5a5" },
    confirm: { bg: "#991b1b", border: "#f87171", text: "#ffffff" },
    sent:    { bg: "#374151", border: "#6b7280", text: "#9ca3af" },
  };
  const c = colors[status];

  return (
    <div style={{ textAlign: "center" }}>
      <button
        onClick={handleClick}
        disabled={status === "sent"}
        style={{
          background: c.bg, border: `2px solid ${c.border}`,
          borderRadius: 12, padding: "16px 40px",
          color: c.text, fontSize: 16, fontWeight: 700,
          cursor: status === "sent" ? "default" : "pointer",
          letterSpacing: 1, transition: "all 0.2s",
        }}
      >
        {status === "idle"    && "🔴  KILL SWITCH"}
        {status === "confirm" && "⚠️  CONFIRM STOP ALL TRADES"}
        {status === "sent"    && "✅  SIGNAL SENT"}
      </button>
      {status === "confirm" && (
        <p style={{ margin: "8px 0 0", fontSize: 12, color: "#ef4444" }}>
          Click again to confirm. This closes all positions immediately.
        </p>
      )}
      {status === "idle" && (
        <p style={{ margin: "8px 0 0", fontSize: 11, color: "#4b5563" }}>
          Emergency stop — closes all open positions
        </p>
      )}
    </div>
  );
}

// ── Trades Table ──────────────────────────────────────────────────────────────

function TradesTable({ trades }: { trades: Trade[] }) {
  if (!trades.length) {
    return (
      <div style={{
        background: "#1a1f2e", border: "1px solid #2a2f42",
        borderRadius: 12, padding: 40, textAlign: "center", color: "#6b7280",
      }}>
        No trades yet. Bot is warming up...
      </div>
    );
  }

  return (
    <div style={{
      background: "#1a1f2e", border: "1px solid #2a2f42",
      borderRadius: 12, overflow: "hidden",
    }}>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ background: "#111827" }}>
              {["Pair","Side","Entry","Exit","PnL %","PnL USDT","Reason","Time"].map(h => (
                <th key={h} style={{
                  padding: "12px 16px", textAlign: "left",
                  color: "#9ca3af", fontWeight: 500,
                  borderBottom: "1px solid #2a2f42", whiteSpace: "nowrap",
                }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {trades.map((t, i) => {
              const profit = parseFloat(t.pnlUsdt) >= 0;
              return (
                <tr key={t.id} style={{
                  background: i % 2 === 0 ? "#1a1f2e" : "#1f2537",
                  borderBottom: "1px solid #2a2f42",
                }}>
                  <td style={{ padding: "10px 16px", color: "#f1f5f9", fontWeight: 600 }}>{t.pair}</td>
                  <td style={{ padding: "10px 16px" }}>
                    <span style={{
                      background: t.side === "buy" ? "#14532d" : "#7f1d1d",
                      color: t.side === "buy" ? "#86efac" : "#fca5a5",
                      padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 700,
                    }}>
                      {t.side.toUpperCase()}
                    </span>
                  </td>
                  <td style={{ padding: "10px 16px", color: "#9ca3af" }}>${t.entryPrice}</td>
                  <td style={{ padding: "10px 16px", color: "#9ca3af" }}>${t.exitPrice}</td>
                  <td style={{ padding: "10px 16px", color: profit ? "#4ade80" : "#f87171", fontWeight: 600 }}>
                    {profit ? "+" : ""}{t.pnlPct}%
                  </td>
                  <td style={{ padding: "10px 16px", color: profit ? "#4ade80" : "#f87171" }}>
                    {profit ? "+" : ""}${t.pnlUsdt}
                  </td>
                  <td style={{ padding: "10px 16px" }}>
                    <span style={{
                      background: t.exitReason === "TAKE_PROFIT" ? "#14532d" :
                                  t.exitReason === "STOP_LOSS"   ? "#7f1d1d" : "#1e3a5f",
                      color: t.exitReason === "TAKE_PROFIT" ? "#86efac" :
                             t.exitReason === "STOP_LOSS"   ? "#fca5a5" : "#93c5fd",
                      padding: "2px 8px", borderRadius: 4, fontSize: 11,
                    }}>
                      {t.exitReason}
                    </span>
                  </td>
                  <td style={{ padding: "10px 16px", color: "#6b7280", fontSize: 12 }}>{t.exitTime}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [stats,       setStats]       = useState<Stats | null>(null);
  const [equity,      setEquity]      = useState<EquityPoint[]>([]);
  const [trades,      setTrades]      = useState<Trade[]>([]);
  const [lastUpdate,  setLastUpdate]  = useState<string>("");
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const res  = await fetch("/api/stats");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      setStats(data.stats);
      setEquity(data.equityCurve);
      setTrades(data.recentTrades);
      setLastUpdate(new Date().toLocaleTimeString());
      setError(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10_000);  // refresh every 10s
    return () => clearInterval(interval);
  }, [fetchData]);

  const pnl      = stats ? parseFloat(stats.totalPnlUsdt) : 0;
  const pnlColor = pnl >= 0 ? "#4ade80" : "#f87171";

  return (
    <div style={{
      background: "#0f1117", minHeight: "100vh",
      fontFamily: "'Inter', 'Segoe UI', sans-serif", color: "#f1f5f9",
    }}>
      {/* ── Header ── */}
      <div style={{
        borderBottom: "1px solid #1f2937",
        padding: "16px 32px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 24 }}>🤖</span>
          <div>
            <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700 }}>TradeBot</h1>
            <p style={{ margin: 0, fontSize: 12, color: "#6b7280" }}>Command Center</p>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
          {/* Live status indicator */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: stats?.inPosition ? "#f59e0b" : "#4ade80",
              boxShadow: `0 0 8px ${stats?.inPosition ? "#f59e0b" : "#4ade80"}`,
            }} />
            <span style={{ fontSize: 13, color: "#9ca3af" }}>
              {stats?.inPosition ? "IN POSITION" : "FLAT"}
            </span>
          </div>

          <span style={{ fontSize: 12, color: "#4b5563" }}>
            Updated: {lastUpdate || "—"}
          </span>
        </div>
      </div>

      <div style={{ padding: "24px 32px", maxWidth: 1400, margin: "0 auto" }}>

        {/* ── Error Banner ── */}
        {error && (
          <div style={{
            background: "#7f1d1d", border: "1px solid #ef4444",
            borderRadius: 8, padding: "12px 16px", marginBottom: 24,
            color: "#fca5a5", fontSize: 14,
          }}>
            ⚠️ {error === "Failed to fetch" || error.includes("ECONNREFUSED")
              ? "Cannot connect to MongoDB. Start the execution engine first."
              : error}
          </div>
        )}

        {loading && !stats ? (
          <div style={{ textAlign: "center", padding: 80, color: "#6b7280" }}>
            Loading dashboard...
          </div>
        ) : (
          <>
            {/* ── Stat Cards Row ── */}
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
              gap: 16, marginBottom: 24,
            }}>
              <StatCard
                label="Balance"
                value={stats ? `$${stats.balance}` : "—"}
                sub="USDT"
              />
              <StatCard
                label="Total PnL"
                value={stats ? `${pnl >= 0 ? "+" : ""}$${stats.totalPnlUsdt}` : "—"}
                color={pnlColor}
                sub="USDT net"
              />
              <StatCard
                label="Win Rate"
                value={stats ? `${stats.winRate}%` : "—"}
                color={parseFloat(stats?.winRate ?? "0") >= 50 ? "#4ade80" : "#f87171"}
              />
              <StatCard
                label="Total Trades"
                value={stats?.totalTrades ?? "—"}
              />
              <StatCard
                label="Profit Factor"
                value={stats?.profitFactor ?? "—"}
                color={parseFloat(stats?.profitFactor ?? "0") >= 1.5 ? "#4ade80" : "#f87171"}
              />
              <StatCard
                label="Avg Win"
                value={stats ? `+${stats.avgWinPct}%` : "—"}
                color="#4ade80"
              />
              <StatCard
                label="Avg Loss"
                value={stats ? `${stats.avgLossPct}%` : "—"}
                color="#f87171"
              />
            </div>

            {/* ── Equity Curve ── */}
            <div style={{
              background: "#1a1f2e", border: "1px solid #2a2f42",
              borderRadius: 12, padding: 24, marginBottom: 24,
            }}>
              <h2 style={{ margin: "0 0 20px", fontSize: 16, fontWeight: 600, color: "#e2e8f0" }}>
                📈 Equity Curve
              </h2>
              {equity.length > 1 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={equity}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2f42" />
                    <XAxis dataKey="time" stroke="#6b7280" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#6b7280" tick={{ fontSize: 11 }}
                           tickFormatter={(v) => `$${v.toFixed(0)}`} />
                    <Tooltip
                      contentStyle={{ background: "#1f2537", border: "1px solid #374151", borderRadius: 8 }}
                      labelStyle={{ color: "#9ca3af" }}
                      formatter={(val: number) => [`$${val.toFixed(2)}`, "Balance"]}
                    />
                    <Line
                      type="monotone" dataKey="balance"
                      stroke="#3b82f6" strokeWidth={2} dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ textAlign: "center", padding: 60, color: "#4b5563" }}>
                  Not enough data for equity curve yet
                </div>
              )}
            </div>

            {/* ── Bottom Row: Trades + Kill Switch ── */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 24 }}>
              {/* Trade History */}
              <div>
                <h2 style={{ margin: "0 0 12px", fontSize: 16, fontWeight: 600 }}>
                  📋 Recent Trades
                </h2>
                <TradesTable trades={trades} />
              </div>

              {/* Kill Switch Panel */}
              <div>
                <h2 style={{ margin: "0 0 12px", fontSize: 16, fontWeight: 600 }}>
                  ⚠️ Emergency Control
                </h2>
                <div style={{
                  background: "#1a1f2e", border: "1px solid #2a2f42",
                  borderRadius: 12, padding: 24,
                }}>
                  <KillSwitch />

                  <div style={{
                    marginTop: 24, padding: 16,
                    background: "#111827", borderRadius: 8,
                    fontSize: 12, color: "#6b7280", lineHeight: 1.6,
                  }}>
                    <strong style={{ color: "#9ca3af" }}>Current Config</strong><br />
                    Pair: {process.env.NEXT_PUBLIC_PAIR || "BTC/USDT"}<br />
                    Max Risk/Trade: {process.env.NEXT_PUBLIC_RISK || "2"}%<br />
                    Stop-Loss: {process.env.NEXT_PUBLIC_SL || "1.5"}%<br />
                    Take-Profit: {process.env.NEXT_PUBLIC_TP || "3"}%
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
