/**
 * indicators.ts
 * ─────────────
 * Computes technical indicators from a candle window in TypeScript.
 * Mirrors feature_engineering.py exactly — same formulas, same order,
 * same normalization — so the model sees identical input at inference
 * time as it did during training.
 *
 * Output: Float32Array of length (windowSize × 18) + 3 portfolio state
 * features, ready to feed directly into onnxruntime.
 */

import { Candle, FeatureVector, CONFIG } from "../utils/types";

// ── Constants (must match Python training) ────────────────────────────────────

const RSI_PERIOD     = 14;
const MACD_FAST      = 12;
const MACD_SLOW      = 26;
const MACD_SIGNAL    = 9;
const BB_PERIOD      = 20;
const BB_STD         = 2.0;
const ATR_PERIOD     = 14;
const EMA_SHORT_A    = 9;
const EMA_SHORT_B    = 21;
const EMA_LONG_A     = 50;
const EMA_LONG_B     = 200;
const VOL_MA_PERIOD  = 20;

// ── EMA ───────────────────────────────────────────────────────────────────────

function computeEMA(values: number[], period: number): number[] {
  const k = 2 / (period + 1);
  const ema: number[] = new Array(values.length).fill(NaN);
  // Seed with SMA of the first `period` values
  const seed = values.slice(0, period).reduce((a, b) => a + b, 0) / period;
  ema[period - 1] = seed;
  for (let i = period; i < values.length; i++) {
    ema[i] = values[i] * k + ema[i - 1] * (1 - k);
  }
  return ema;
}

function computeEMAFromFirstValid(values: number[], period: number): number[] {
  const k = 2 / (period + 1);
  const ema: number[] = new Array(values.length).fill(NaN);
  const firstValid = values.findIndex(v => !isNaN(v));
  if (firstValid === -1 || values.length - firstValid < period) return ema;

  const seedEnd = firstValid + period;
  const seed = values
    .slice(firstValid, seedEnd)
    .reduce((a, b) => a + b, 0) / period;

  ema[seedEnd - 1] = seed;
  for (let i = seedEnd; i < values.length; i++) {
    if (isNaN(values[i])) {
      ema[i] = ema[i - 1];
    } else {
      ema[i] = values[i] * k + ema[i - 1] * (1 - k);
    }
  }
  return ema;
}

// ── RSI ───────────────────────────────────────────────────────────────────────

function computeRSI(closes: number[], period: number): number[] {
  const rsi: number[] = new Array(closes.length).fill(NaN);
  if (closes.length < period + 1) return rsi;

  let avgGain = 0, avgLoss = 0;
  for (let i = 1; i <= period; i++) {
    const diff = closes[i] - closes[i - 1];
    if (diff >= 0) avgGain += diff;
    else avgLoss += Math.abs(diff);
  }
  avgGain /= period;
  avgLoss /= period;

  for (let i = period; i < closes.length; i++) {
    if (i > period) {
      const diff = closes[i] - closes[i - 1];
      const gain = diff >= 0 ? diff : 0;
      const loss = diff < 0 ? Math.abs(diff) : 0;
      avgGain = (avgGain * (period - 1) + gain) / period;
      avgLoss = (avgLoss * (period - 1) + loss) / period;
    }
    const rs  = avgLoss === 0 ? 100 : avgGain / avgLoss;
    rsi[i]    = 100 - 100 / (1 + rs);
  }
  return rsi;
}

// ── ATR ───────────────────────────────────────────────────────────────────────

function computeATR(candles: Candle[], period: number): number[] {
  const tr: number[]  = new Array(candles.length).fill(0);
  const atr: number[] = new Array(candles.length).fill(NaN);

  for (let i = 1; i < candles.length; i++) {
    const h = candles[i].high;
    const l = candles[i].low;
    const pc = candles[i - 1].close;
    tr[i] = Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc));
  }

  // Seed ATR with SMA of first period TR values
  let sum = 0;
  for (let i = 1; i <= period; i++) sum += tr[i];
  atr[period] = sum / period;

  for (let i = period + 1; i < candles.length; i++) {
    atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period;
  }
  return atr;
}

// ── Bollinger Bands ───────────────────────────────────────────────────────────

function computeBollinger(closes: number[], period: number, stdDev: number) {
  const mid:   number[] = new Array(closes.length).fill(NaN);
  const upper: number[] = new Array(closes.length).fill(NaN);
  const lower: number[] = new Array(closes.length).fill(NaN);

  for (let i = period - 1; i < closes.length; i++) {
    const slice = closes.slice(i - period + 1, i + 1);
    const mean  = slice.reduce((a, b) => a + b, 0) / period;
    const std   = Math.sqrt(slice.reduce((s, v) => s + (v - mean) ** 2, 0) / period);
    mid[i]   = mean;
    upper[i] = mean + stdDev * std;
    lower[i] = mean - stdDev * std;
  }
  return { mid, upper, lower };
}

// ── ADX ───────────────────────────────────────────────────────────────────────

function computeADX(candles: Candle[], period: number): number[] {
  const n = candles.length;
  const tr: number[] = new Array(n).fill(0);
  const pdm: number[] = new Array(n).fill(0);
  const ndm: number[] = new Array(n).fill(0);
  for (let i = 1; i < n; i++) {
    const h = candles[i].high, l = candles[i].low, pc = candles[i - 1].close;
    tr[i] = Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc));
    const upMove = h - candles[i - 1].high;
    const downMove = candles[i - 1].low - l;
    pdm[i] = (upMove > downMove && upMove > 0) ? upMove : 0;
    ndm[i] = (downMove > upMove && downMove > 0) ? downMove : 0;
  }
  
  const smooth = (values: number[], period: number) => {
    const res: number[] = new Array(n).fill(NaN);
    let sum = 0;
    for (let i = 1; i <= period; i++) sum += values[i];
    res[period] = sum;
    for (let i = period + 1; i < n; i++) res[i] = res[i - 1] - (res[i - 1] / period) + values[i];
    return res;
  };

  const str = smooth(tr, period);
  const spdm = smooth(pdm, period);
  const sndm = smooth(ndm, period);

  const dx: number[] = new Array(n).fill(NaN);
  for (let i = period; i < n; i++) {
    const diPlus = 100 * (spdm[i] / (str[i] || 1));
    const diMinus = 100 * (sndm[i] / (str[i] || 1));
    const diff = Math.abs(diPlus - diMinus);
    const sum = diPlus + diMinus;
    dx[i] = sum === 0 ? 0 : 100 * (diff / sum);
  }

  const adx: number[] = new Array(n).fill(NaN);
  let dxSum = 0;
  for (let i = period; i < period * 2; i++) dxSum += dx[i];
  adx[period * 2 - 1] = dxSum / period;
  
  for (let i = period * 2; i < n; i++) {
    adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period;
  }
  return adx;
}

// ── OBV ───────────────────────────────────────────────────────────────────────

function computeOBV(candles: Candle[]): number[] {
  const obv: number[] = new Array(candles.length).fill(0);
  let current = 0;
  for (let i = 1; i < candles.length; i++) {
    if (candles[i].close > candles[i - 1].close) current += candles[i].volume;
    else if (candles[i].close < candles[i - 1].close) current -= candles[i].volume;
    obv[i] = current;
  }
  return obv;
}

// ── Core Feature Builder ──────────────────────────────────────────────────────

/**
 * Takes a window of candles and returns a flat Float32Array ready
 * for onnxruntime. Shape: [1, windowSize × 18 + 3]
 */
/**
 * Live portfolio state — must mirror trading_env._get_observation's seven
 * appended features EXACTLY (order, normalization, caps). The smoke test
 * caught the engine sending the legacy 3-feature tail (obs 1539 vs 1543).
 */
export interface PortfolioState {
  positionHeld:       boolean;
  entryPrice:         number;  // 0 when flat
  positionSizeBtc:    number;  // BTC units held, 0 when flat
  stepsInPosition:    number;  // candles since entry, 0 when flat
  stepsSinceTrade:    number;  // candles since last close (or process start)
  balance:            number;  // free cash (USDT)
  peakPortfolioValue: number;  // running max of mark-to-market equity
  initialBalance:     number;  // CONFIG.initialBalance
}

export function buildObservationTensor(
  candles: Candle[],
  ps:      PortfolioState,
): Float32Array {
  const n = candles.length;
  if (n < EMA_LONG_B) {
    throw new Error(`Need at least ${EMA_LONG_B} candles, got ${n}`);
  }

  const closes  = candles.map(c => c.close);
  const highs   = candles.map(c => c.high);
  const lows    = candles.map(c => c.low);
  const volumes = candles.map(c => c.volume);

  // ── Pre-compute all series ───────────────────────────────────────────────
  const rsiArr  = computeRSI(closes, RSI_PERIOD);

  const emaFast    = computeEMA(closes, MACD_FAST);
  const emaSlow    = computeEMA(closes, MACD_SLOW);
  const macdLine   = emaFast.map((v, i) => isNaN(v) || isNaN(emaSlow[i]) ? NaN : v - emaSlow[i]);
  const macdSig    = computeEMAFromFirstValid(macdLine, MACD_SIGNAL);

  const bb = computeBollinger(closes, BB_PERIOD, BB_STD);
  const atrArr = computeATR(candles, ATR_PERIOD);

  const ema9   = computeEMA(closes, EMA_SHORT_A);
  const ema21  = computeEMA(closes, EMA_SHORT_B);
  const ema50  = computeEMA(closes, EMA_LONG_A);
  const ema200 = computeEMA(closes, EMA_LONG_B);

  const volMA: number[] = new Array(n).fill(NaN);
  for (let i = VOL_MA_PERIOD - 1; i < n; i++) {
    const slice = volumes.slice(i - VOL_MA_PERIOD + 1, i + 1);
    volMA[i] = slice.reduce((a, b) => a + b, 0) / VOL_MA_PERIOD;
  }

  const adxArr = computeADX(candles, 14);
  const obvArr = computeOBV(candles);
  
  // MTF Proxies
  const rsi4hArr = computeRSI(closes, 14 * 16);
  const emaFast4h = computeEMA(closes, 12 * 16);
  const emaSlow4h = computeEMA(closes, 26 * 16);
  const rsi1dArr = computeRSI(closes, 14 * 96);
  const emaFast1d = computeEMA(closes, 12 * 96);
  const emaSlow1d = computeEMA(closes, 26 * 96);

  // ── Build feature matrix for the last windowSize candles ──────────────────
  const window    = candles.slice(-CONFIG.windowSize);
  const wStart    = n - CONFIG.windowSize;
  const features: number[] = [];

  for (let wi = 0; wi < CONFIG.windowSize; wi++) {
    const i  = wStart + wi;
    const c  = candles[i];
    const p  = closes[i];

    // Log returns
    const logR  = i > 0 ? Math.log(closes[i]  / closes[i - 1])  : 0;
    const logRH = i > 0 ? Math.log(highs[i]   / highs[i - 1])   : 0;
    const logRL = i > 0 ? Math.log(lows[i]    / lows[i - 1])    : 0;
    const v0 = Math.max(volumes[i],     1e-8);
    const v1 = Math.max(volumes[i - 1] ?? 1e-8, 1e-8);
    const logRV = i > 0 ? Math.log(v0 / v1) : 0;

    // Candle structure
    const range   = c.high - c.low || 1e-8;
    const bodyRatio      = Math.abs(c.close - c.open)                         / range;
    const upperWickRatio = (c.high - Math.max(c.open, c.close))               / range;
    const lowerWickRatio = (Math.min(c.open, c.close) - c.low)                / range;
    const candleDir      = Math.sign(c.close - c.open);

    // RSI: scale [0,100] → [-1,1]
    const rsi = isNaN(rsiArr[i]) ? 0 : (rsiArr[i] - 50) / 50;

    // MACD (normalized by rolling mean price)
    const priceMa = closes.slice(Math.max(0, i - MACD_SLOW), i + 1)
                          .reduce((a, b) => a + b, 0) /
                    Math.min(i + 1, MACD_SLOW) || 1;
    const macd       = isNaN(macdLine[i]) ? 0 : macdLine[i] / priceMa;
    const macdSigVal = isNaN(macdSig[i])  ? 0 : macdSig[i]  / priceMa;
    const macdHist   = macd - macdSigVal;

    // Bollinger position
    const bbMid   = bb.mid[i]   || p;
    const bbUpper = bb.upper[i] || p;
    const bbLower = bb.lower[i] || p;
    const bbBand  = (bbUpper - bbLower) / 2 || 1e-8;
    const bbPos   = (p - bbMid) / bbBand;
    const bbWidth = (bbUpper - bbLower) / (bbMid || 1);

    // ATR ratio
    const atrRatio = isNaN(atrArr[i]) ? 0 : atrArr[i] / (p || 1);

    // EMA crossovers
    const emaCrossShort = (isNaN(ema9[i]) || isNaN(ema21[i])) ? 0
                          : (ema9[i] - ema21[i]) / (p || 1);
    const emaCrossLong  = (isNaN(ema50[i]) || isNaN(ema200[i])) ? 0
                          : (ema50[i] - ema200[i]) / (p || 1);

    // Volume ratio
    const volRatio = isNaN(volMA[i]) ? 1 : volumes[i] / (volMA[i] || 1);

    // v2: Time Encoding
    const d = new Date(c.timestamp);
    const hour = d.getUTCHours();
    const dow = d.getUTCDay();
    const hourSin = Math.sin(2 * Math.PI * hour / 24);
    const hourCos = Math.cos(2 * Math.PI * hour / 24);
    const daySin = Math.sin(2 * Math.PI * dow / 7);
    const dayCos = Math.cos(2 * Math.PI * dow / 7);

    // v2: ADX
    const adxRaw = isNaN(adxArr[i]) ? 25 : adxArr[i];
    const adx = Math.max(-1, Math.min(1, (adxRaw - 25) / 25));

    // v2: OBV Ratio
    const obvWindow = 20;
    const obvSlice = obvArr.slice(Math.max(0, i - obvWindow + 1), i + 1);
    const obvMean = obvSlice.reduce((a, b) => a + b, 0) / obvSlice.length;
    const obvStd = Math.sqrt(obvSlice.reduce((s, v) => s + (v - obvMean) ** 2, 0) / obvSlice.length) || 1e-8;
    const obvRatio = Math.max(-1, Math.min(1, ((obvArr[i] - obvMean) / obvStd) / 3));

    // v3: MTF
    const rsi4h = isNaN(rsi4hArr[i]) ? 0 : (rsi4hArr[i] - 50) / 50;
    const rsi1d = isNaN(rsi1dArr[i]) ? 0 : (rsi1dArr[i] - 50) / 50;
    
    const priceMa4h = closes.slice(Math.max(0, i - 26*16 + 1), i + 1).reduce((a,b)=>a+b,0) / Math.min(i+1, 26*16) || 1;
    const macd4h = isNaN(emaFast4h[i]) || isNaN(emaSlow4h[i]) ? 0 : (emaFast4h[i] - emaSlow4h[i]) / priceMa4h;

    const priceMa1d = closes.slice(Math.max(0, i - 26*96 + 1), i + 1).reduce((a,b)=>a+b,0) / Math.min(i+1, 26*96) || 1;
    const macd1d = isNaN(emaFast1d[i]) || isNaN(emaSlow1d[i]) ? 0 : (emaFast1d[i] - emaSlow1d[i]) / priceMa1d;

    // v4: Macro
    const macroWin = 2880; // 30 days
    const maxH = Math.max(...highs.slice(Math.max(0, i - macroWin + 1), i + 1));
    const distFromHigh = (p - maxH) / maxH;
    
    const macroSlice = closes.slice(Math.max(0, i - macroWin + 1), i + 1);
    const macroMean = macroSlice.reduce((a, b) => a + b, 0) / macroSlice.length;
    const macroTrendSma = (p - macroMean) / macroMean;
    
    let sumRet = 0;
    for(let j = Math.max(1, i - macroWin + 1); j <= i; j++) {
      sumRet += (closes[j] - closes[j-1]) / closes[j-1];
    }
    const meanRet = sumRet / Math.min(i, macroWin) || 0;
    let sumSqRet = 0;
    for(let j = Math.max(1, i - macroWin + 1); j <= i; j++) {
      sumSqRet += ((closes[j] - closes[j-1]) / closes[j-1] - meanRet) ** 2;
    }
    const macroVol = Math.sqrt(sumSqRet / Math.min(i, macroWin) || 0) * Math.sqrt(96 * 365);
    
    const obvMacroSlice = obvArr.slice(Math.max(0, i - macroWin + 1), i + 1);
    const obvMacroMean = obvMacroSlice.reduce((a, b) => a + b, 0) / obvMacroSlice.length;
    const obvMacroStd = Math.sqrt(obvMacroSlice.reduce((s, v) => s + (v - obvMacroMean) ** 2, 0) / obvMacroSlice.length) || 1e-8;
    const macroObvRatio = Math.max(-1, Math.min(1, ((obvArr[i] - obvMacroMean) / obvMacroStd) / 3));

    features.push(
      logR, logRH, logRL, logRV,
      bodyRatio, upperWickRatio, lowerWickRatio, candleDir,
      rsi,
      macd, macdSigVal, macdHist,
      bbPos, bbWidth,
      atrRatio,
      emaCrossShort, emaCrossLong,
      volRatio,
      hourSin, hourCos, daySin, dayCos,
      adx, obvRatio,
      rsi4h, macd4h, rsi1d, macd1d,
      distFromHigh, macroTrendSma, macroVol, macroObvRatio
    );
  }

  // ── Portfolio state (7 values, mirroring trading_env._get_observation) ───
  const latestPrice = closes[n - 1];
  const windowSize  = CONFIG.windowSize;

  // mark-to-market equity = cash + open position value (env: _get_portfolio_value)
  const portfolioValue = ps.balance +
    (ps.positionHeld ? ps.positionSizeBtc * latestPrice : 0);

  const posHeld     = ps.positionHeld ? 1.0 : 0.0;
  const unrealized  = ps.positionHeld
                      ? (latestPrice - ps.entryPrice) / (ps.entryPrice + 1e-8)
                      : 0.0;
  const drawdown    = (ps.peakPortfolioValue - portfolioValue) /
                      (ps.peakPortfolioValue + 1e-8);
  const stepsInPos  = ps.positionHeld ? ps.stepsInPosition / windowSize : 0.0;
  const portReturn  = portfolioValue / ps.initialBalance - 1.0;
  const posSize     = ps.positionHeld && portfolioValue > 0
                      ? (ps.positionSizeBtc * latestPrice) / portfolioValue
                      : 0.0;
  const stepsSince  = Math.min(ps.stepsSinceTrade / windowSize, 5.0);

  features.push(posHeld, unrealized, drawdown, stepsInPos,
                portReturn, posSize, stepsSince);

  return new Float32Array(features);
}
