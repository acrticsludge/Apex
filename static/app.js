const INTERVAL = 30;
let _progStart = null, _progTimer = null, _refreshTimer = null;
let _cfg = {};        // latest config snapshot — set in doRefresh
let _lastState = null; // last full /api/state response for SSE PnL recalc
let _activeTab = "india";
let _logPollTimer = null;
let _sessPollTimer = null;
let _agentPollTimer = null;
let _retrainPollTimer = null;
let _logFilter = "ALL";
let _autoScroll = true;
let _lastLogEntries = null;
let _lastDlogEntries = [];
let _dlogFilter = "ALL";
let _sessFilter = "all";
let _sessData   = [];

// ── Utilities ──────────────────────────────────────────────────────────────

// Format a UTC ISO timestamp string into local date + time.
// Always shows "15 Apr 14:30:22" so the date is visible in the log.
function fmtTs(ts) {
  if (!ts || ts === "—") return ts;
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts;          // graceful fallback for legacy strings
  const timePart = d.toLocaleTimeString(undefined, {hour:"2-digit", minute:"2-digit", second:"2-digit", hour12:false});
  const datePart = d.toLocaleDateString(undefined, {day:"2-digit", month:"short"});
  return datePart + "  " + timePart;
}

// Full date + time for the clock / last-update label.
function fmtDt(ts) {
  const d = ts ? new Date(ts) : new Date();
  if (isNaN(d.getTime())) return ts;
  return d.toLocaleDateString(undefined, {day:"2-digit", month:"short", year:"numeric"}) + "  " +
         d.toLocaleTimeString(undefined, {hour:"2-digit", minute:"2-digit", second:"2-digit", hour12:false});
}

function fc(v, sym) {
  if (v == null) return "—";
  const a = Math.abs(v);
  if (sym === "₹") {
    if (a >= 1e7) return sym + (v/1e7).toFixed(2) + "Cr";
    if (a >= 1e5) return sym + (v/1e5).toFixed(2) + "L";
  }
  if (a >= 1e6) return sym + (v/1e6).toFixed(2) + "M";
  if (a >= 1e3) return sym + Math.abs(v).toLocaleString("en-IN",{maximumFractionDigits:0});
  return sym + v.toFixed(2);
}
const sc  = v => v > 0 ? "green" : v < 0 ? "red" : "muted";
const sgn = v => v >= 0 ? "+" : "";
const disp = s => s.replace(".NS","");

function toast(msg, ok=true) {
  const d = document.createElement("div");
  d.className = "toast " + (ok ? "tok" : "terr");
  d.innerHTML = `<span>${ok?"✓":"✗"}</span>${msg}`;
  document.getElementById("toasts").appendChild(d);
  setTimeout(() => d.remove(), 3200);
}

function tab(name, el) {
  _activeTab = name;
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("on"));
  document.querySelectorAll(".pane").forEach(p => p.classList.remove("on"));
  el.classList.add("on");
  document.getElementById("pane-"+name).classList.add("on");
  clearInterval(_logPollTimer);
  clearInterval(_sessPollTimer);
  clearInterval(_agentPollTimer);
  clearInterval(_retrainPollTimer);
  if (name === "logs") {
    _logPollTimer = setInterval(pollLogs, 5000);
    setInterval(pollDecisions, 4000);
    pollLogs();
    pollDecisions();
  } else if (name === "sessions") {
    _sessPollTimer = setInterval(pollSessions, 15000);
    pollSessions();
  } else if (name === "agent") {
    _agentPollTimer = setInterval(pollAgentTerminal, 4000);
    pollAgentTerminal();
  } else if (name === "retrain") {
    _retrainPollTimer = setInterval(pollRetrainTerminal, 5000);
    pollRetrainTerminal();
  }
}

async function pollLogs() {
  try {
    const r = await fetch("/api/logs");
    if (!r.ok) return;
    const entries = await r.json();
    _lastLogEntries = entries;
    renderAgentLog(entries);
    const dot = document.getElementById("log-live-dot");
    if (dot) { dot.style.color = "var(--green)"; setTimeout(()=>{ dot.style.color="var(--muted)"; }, 800); }
  } catch(e) {}
}

function setLogFilter(lvl, el) {
  _logFilter = lvl;
  document.querySelectorAll(".log-filter-btn").forEach(b => b.classList.remove("on"));
  el.classList.add("on");
  if (_lastLogEntries) renderAgentLog(_lastLogEntries);
}

// ── Decision log ──────────────────────────────────────────────────────────
async function pollDecisions() {
  try {
    const r = await fetch("/api/think");
    if (!r.ok) return;
    const entries = await r.json();
    _lastDlogEntries = entries;
    renderDlog(entries);
    const dot = document.getElementById("dlog-live-dot");
    if (dot) { dot.style.color = "var(--green)"; setTimeout(()=>{ dot.style.color="var(--muted)"; }, 800); }
  } catch(e) {}
}

// ── Agent Terminal ────────────────────────────────────────────────────────
const _RL_COLORS = {BUY:'#3fb950', SELL:'#f85149', HOLD:'#8b949e', WARNING:'#f0883e', online:'#3fb950', Cycle:'#58a6ff'};

function renderAgentTerminal(entries) {
  const el = document.getElementById('agent-terminal');
  if (!el) return;
  const badge = document.getElementById('rl-status-badge');
  if (!entries || !entries.length) {
    el.innerHTML = '<span style="color:#484f58">—</span> <span style="color:#666">No RL decisions yet — start the agent</span>';
    if (badge) { badge.textContent = '● INACTIVE'; badge.style.color = '#f85149'; }
    return;
  }
  const isActive = entries.some(e => Date.now() - new Date(e.ts).getTime() < 600000);
  if (badge) {
    badge.textContent = isActive ? '● ACTIVE' : '● INACTIVE';
    badge.style.color  = isActive ? '#3fb950' : '#f85149';
    badge.style.background = isActive ? '#0a2016' : '#2a0a0a';
  }
  el.innerHTML = entries.map(e => {
    const colorKey = Object.keys(_RL_COLORS).find(k => e.msg && e.msg.includes(k));
    const c = colorKey ? _RL_COLORS[colorKey] : '#58a6ff';
    const ts = e.ts ? new Date(e.ts).toLocaleString(undefined,{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).replace(',','') : '';
    const sym = (e.sym && e.sym !== 'SYSTEM') ? `<span style="color:#58a6ff;margin-right:6px">${e.sym}</span>` : '';
    return `<div style="margin-bottom:3px;border-bottom:1px solid #161b22;padding-bottom:3px">` +
      `<span style="color:#484f58">${ts}</span>` +
      `<span style="color:#8b949e;margin:0 6px;font-size:10px">[RL]</span>` +
      sym +
      `<span style="color:${c}">${e.msg || ''}</span></div>`;
  }).join('');
  const auto = document.getElementById('agent-autoscroll');
  if (auto?.checked) el.scrollTop = el.scrollHeight;
}

async function pollAgentTerminal() {
  try {
    const r = await fetch('/api/think');
    if (!r.ok) return;
    const entries = await r.json();
    const rl = entries.filter(e => e.cat === 'RL');
    renderAgentTerminal(rl);
    const dot = document.getElementById('agent-live-dot');
    if (dot) { dot.style.color='var(--green)'; setTimeout(()=>{ dot.style.color='var(--muted)'; }, 800); }
  } catch(e) {}
}

function clearAgentTerminal() {
  fetch('/api/think/clear', {method:'POST'}).catch(()=>{});
  const el = document.getElementById('agent-terminal');
  if (el) el.innerHTML = '<span style="color:#484f58">—</span> <span style="color:#666">Cleared — waiting for next cycle</span>';
  _lastDlogEntries = [];
  const dbox = document.getElementById("dlog-box");
  if (dbox) renderDlog([]);
}

// ── Retraining Terminal ───────────────────────────────────────────────────
function renderRetrainTerminal(data) {
  const el      = document.getElementById('retrain-terminal');
  const badge   = document.getElementById('retrain-status-badge');
  const bufCnt  = document.getElementById('retrain-buf-count');
  const bufFill = document.getElementById('retrain-buf-fill');
  const total   = document.getElementById('retrain-total');
  const pending = document.getElementById('retrain-pending');
  if (!el) return;

  const thr  = data.threshold || 16;
  const nc   = data.new_count  || 0;
  const bufsz= data.buffer_size || 0;

  if (bufCnt)  bufCnt.textContent  = nc;
  if (bufFill) bufFill.style.width = Math.min(100, Math.round(nc / thr * 100)) + '%';
  if (total)   total.textContent   = data.total_updates || 0;
  if (pending) pending.textContent = bufsz;

  if (badge) {
    if (data.is_training) {
      badge.textContent = '⟳ TRAINING'; badge.style.color = '#f0883e'; badge.style.background = '#2a1800';
    } else if ((data.total_updates || 0) > 0) {
      badge.textContent = '● UPDATED';  badge.style.color = '#3fb950'; badge.style.background = '#0a2016';
    } else {
      badge.textContent = '● IDLE';     badge.style.color = '#8b949e'; badge.style.background = '#1a1a1a';
    }
  }

  const entries = data.log || [];
  if (!entries.length) {
    el.innerHTML = '<span style="color:#484f58">—</span> <span style="color:#666">No retraining events yet — waiting for 16 closed RL trades to accumulate</span>';
    return;
  }

  el.innerHTML = [...entries].reverse().map(e => {
    const ts = e.ts ? new Date(e.ts).toLocaleString(undefined,
      {month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}
    ).replace(',','') : '';

    let color, prefix, icon;
    if (e.event === 'started') {
      color = '#58a6ff'; prefix = 'STARTED'; icon = '🔄';
    } else if (e.event === 'completed') {
      color = (e.improvement_pct || 0) >= 0 ? '#3fb950' : '#f0883e';
      prefix = 'APPLIED'; icon = '✓';
    } else if (e.event === 'failed') {
      color = '#f85149'; prefix = 'FAILED'; icon = '✗';
    } else if (e.event === 'skipped') {
      color = '#8b949e'; prefix = 'SKIPPED'; icon = '○';
    } else {
      color = '#8b949e'; prefix = (e.event || '').toUpperCase(); icon = '·';
    }

    const tradesTag = e.trades
      ? `<span style="color:#484f58;font-size:10px;margin-left:8px">${e.trades} trades</span>` : '';
    const improvTag = e.improvement_pct !== undefined
      ? `<span style="color:${(e.improvement_pct||0)>=0?'#3fb950':'#f85149'};font-size:10px;margin-left:8px">${e.improvement_pct>0?'+':''}${e.improvement_pct}%</span>` : '';
    const errTag = e.error
      ? `<div style="color:#f85149;font-size:10px;margin-left:16px;margin-top:2px">${e.error}</div>` : '';

    return `<div style="margin-bottom:5px;border-bottom:1px solid #161b22;padding-bottom:5px">` +
      `<span style="color:#484f58">${ts}</span>` +
      `<span style="color:${color};margin:0 7px;font-weight:700">${icon} ${prefix}</span>` +
      `<span style="color:${color}">${e.msg || ''}</span>` +
      tradesTag + improvTag + errTag +
      `</div>`;
  }).join('');
}

async function pollRetrainTerminal() {
  try {
    const r = await fetch('/api/retrain/log');
    if (!r.ok) return;
    renderRetrainTerminal(await r.json());
  } catch(e) {}
}

function setDlogFilter(cat, el) {
  _dlogFilter = cat;
  document.querySelectorAll('[id^="df-"]').forEach(b => b.classList.remove("on"));
  if (el) el.classList.add("on");
  renderDlog(_lastDlogEntries);
}

function renderDlog(entries) {
  const box = document.getElementById("dlog-box");
  const cnt = document.getElementById("dlog-count");
  if (!box) return;
  const all = entries || [];
  const filtered = _dlogFilter === "ALL" ? all : all.filter(e => e.cat === _dlogFilter);
  if (cnt) cnt.textContent = filtered.length + (filtered.length < all.length ? " / "+all.length : "") + " entries";
  if (!filtered.length) {
    box.innerHTML = `<div class="drow"><span class="al-ts">—</span><span class="dcat dcat-CYCLE">CYCLE</span><span class="dsym"></span><span class="dmsg muted">No ${_dlogFilter === "ALL" ? "" : _dlogFilter+" "}decisions yet</span></div>`;
    return;
  }
  box.innerHTML = [...filtered].reverse().map(e => {
    const cat = e.cat || "CYCLE";
    const ts  = fmtTs(e.ts || "");
    const sym = e.sym ? `<span class="dsym">${e.sym}</span>` : `<span class="dsym"></span>`;
    return `<div class="drow">
      <span class="al-ts">${ts}</span>
      <span class="dcat dcat-${cat}">${cat}</span>
      ${sym}
      <span class="dmsg">${e.msg || ""}</span>
    </div>`;
  }).join("");
  const auto = document.getElementById("dlog-autoscroll");
  if (auto?.checked) box.scrollTop = 0;
}

// ── Clock (browser local time) ────────────────────────────────────────────
function tick() {
  document.getElementById("clock").textContent = fmtDt();
}
setInterval(tick, 1000); tick();

// ── Progress bar ──────────────────────────────────────────────────────────
function startProg() {
  _progStart = Date.now();
  clearInterval(_progTimer);
  _progTimer = setInterval(() => {
    const e = (Date.now()-_progStart)/1000;
    document.getElementById("prog-f").style.width = Math.min(100, e/INTERVAL*100)+"%";
    const r = Math.max(0, INTERVAL - Math.floor(e));
    document.getElementById("nxt-ref").textContent = "Refresh in "+r+"s";
  }, 500);
}

// ── Render stats cards ────────────────────────────────────────────────────
function renderStats(d) {
  const ic = d.config.india_capital, uc = d.config.us_capital;
  [["india","₹",ic],["us","$",uc]].forEach(([mkt,sym,cap]) => {
    const m  = d[mkt];
    const pv = m.portfolio_value || 0;
    const df = pv - cap, pct = df/cap*100;
    const pnl = m.realised_pnl || 0;
    const tot = (m.wins||0)+(m.losses||0);
    const wr  = tot > 0 ? m.wins/tot*100 : 0;
    const dd   = m.drawdown || 0;
    const tpnl = m.total_pnl || 0;
    const upnl = m.unrealised_pnl || 0;
    document.getElementById(mkt+"-pv").textContent   = fc(pv,sym);
    document.getElementById(mkt+"-pv").className     = "c-val mono "+sc(df);
    document.getElementById(mkt+"-diff").innerHTML   = `<span class="${sc(df)}">${sgn(df)}${fc(df,sym)} (${sgn(pct)}${pct.toFixed(1)}%)</span>`;
    document.getElementById(mkt+"-cash").textContent = fc(m.cash,sym);
    document.getElementById(mkt+"-npos").textContent = Object.keys(m.positions||{}).length+" positions";
    document.getElementById(mkt+"-pnl").textContent  = fc(pnl,sym);
    document.getElementById(mkt+"-pnl").className    = "c-val mono "+sc(pnl);
    document.getElementById(mkt+"-ntrades").textContent = tot+" trades";
    document.getElementById(mkt+"-wr").textContent   = wr.toFixed(0)+"%";
    document.getElementById(mkt+"-wr").className     = "c-val mono "+(wr>=50?"green":"red");
    document.getElementById(mkt+"-wl").textContent   = (m.wins||0)+"W "+(m.losses||0)+"L";
    document.getElementById(mkt+"-dd").textContent   = dd.toFixed(1)+"%";
    document.getElementById(mkt+"-dd").className     = "c-val mono "+(dd>5?"red":"green");
    document.getElementById(mkt+"-tpnl").textContent = (tpnl>=0?"+":"")+fc(tpnl,sym);
    document.getElementById(mkt+"-tpnl").className   = "c-val mono "+sc(tpnl);
    document.getElementById(mkt+"-upnl").innerHTML   =
      `Float: <span class="${sc(upnl)}">${sgn(upnl)}${fc(upnl,sym)}</span>`;
    // market open badge
    const mb = document.getElementById(mkt+"-mkt");
    mb.className = "badge "+(m.market_open?"b-open":"b-closed");
    mb.textContent = m.market_open ? "OPEN" : "CLOSED";
    // session info
    const sl = document.getElementById(mkt+"-sess-lbl");
    if (sl) {
      const sessDate = m.session_date || "—";
      const nTrades  = (m.trade_log||[]).length;
      const pnlStr   = (m.realised_pnl||0) >= 0
        ? "+" + fc(m.realised_pnl||0, sym) : fc(m.realised_pnl||0, sym);
      sl.innerHTML = `Session <span class="mono" style="color:var(--muted)">${sessDate}</span>`
        + `  ·  ${nTrades} trade${nTrades!==1?"s":""}  ·  `
        + `<span class="${sc(m.realised_pnl||0)}">${pnlStr}</span>`;
    }
  });
  // header badges
  const nb = document.getElementById("nse-badge"), ub = document.getElementById("nyse-badge");
  nb.className = "badge "+(d.india.market_open?"b-open":"b-closed");
  nb.textContent = "NSE "+(d.india.market_open?"OPEN":"CLOSED");
  ub.className = "badge "+(d.us.market_open?"b-open":"b-closed");
  ub.textContent = "NYSE "+(d.us.market_open?"OPEN":"CLOSED");
}

// ── EOD per-position cell ─────────────────────────────────────────────────
function eodCell(p, cur, sym, mtc) {
  if (mtc == null || mtc <= 0) return `<td class="muted" style="font-size:10px">—</td>`;
  const harvestMin = _cfg.eod_harvest_min || 30;
  const exitMin    = _cfg.eod_exit_min    || 15;
  const isShort    = p.side === "short";
  const pnl        = isShort ? (p.entry - cur) * p.qty : (cur - p.entry) * p.qty;
  const needPct    = isShort
    ? ((cur / p.entry) - 1) * 100   // % drop needed to break even for shorts
    : ((p.entry / cur) - 1) * 100;  // % rise needed to break even for longs
  let badge = "", note = "";

  if (mtc <= exitMin) {
    badge = `<span style="color:var(--red);font-size:10px;font-weight:700">⚡ FORCE EXIT</span>`;
  } else if (mtc <= harvestMin) {
    if (pnl > 0) {
      badge = `<span style="color:var(--green);font-size:10px;font-weight:700">✓ HARVESTING</span>`;
    } else {
      badge = `<span style="color:var(--orange);font-size:10px;font-weight:700">⏳ HOLDING</span>`;
      if (needPct > 0.05)
        note = `<div style="color:var(--yellow);font-size:9px">need +${needPct.toFixed(1)}% → ${sym}${p.entry.toFixed(2)} BE</div>`;
    }
  } else {
    const m = Math.floor(mtc);
    badge = `<span style="color:var(--muted);font-size:10px">${m}m left</span>`;
    if (pnl < 0 && mtc < 60 && needPct > 0.05)
      note = `<div style="color:var(--yellow);font-size:9px">+${needPct.toFixed(1)}% to BE</div>`;
  }
  return `<td>${badge}${note}</td>`;
}

// ── Render positions table ────────────────────────────────────────────────
function renderPos(positions, prices, sym, bodyId, countId, mkt, mtc) {
  const cnt = Object.keys(positions||{}).length;
  document.getElementById(countId).textContent = cnt;
  const tb = document.getElementById(bodyId);
  if (!cnt) {
    tb.innerHTML = `<tr><td colspan="10" style="text-align:center;padding:18px;color:var(--muted)">No open positions</td></tr>`;
    return;
  }
  tb.innerHTML = Object.entries(positions).map(([s,p]) => {
    const cur      = prices[s] || p.entry;
    const isShort  = p.side === "short";
    const pnl      = isShort ? (p.entry - cur) * p.qty : (cur - p.entry) * p.qty;
    const pct      = pnl / (p.entry * p.qty) * 100;
    const c        = pnl >= 0 ? "green" : "red";
    const sideBadge = `<span class="side-badge ${isShort ? 'side-short' : 'side-long'}">${isShort ? 'SHORT' : 'LONG'}</span>`;
    const btnLabel  = isShort ? "Cover" : "Sell";
    return `<tr>
      <td class="mono" style="font-weight:700;color:var(--blue)">${disp(s)} ${sideBadge}</td>
      <td class="mono">${p.qty}</td>
      <td class="mono">${sym}${p.entry.toFixed(2)}</td>
      <td class="mono">${sym}${cur.toFixed(2)}</td>
      <td class="mono ${c}">${pnl>=0?"+":""}${sym}${Math.abs(pnl).toFixed(0)} (${pct>=0?"+":""}${pct.toFixed(1)}%)</td>
      <td class="mono">${sym}${p.stop_loss.toFixed(2)}</td>
      <td class="mono">${sym}${p.target.toFixed(2)}</td>
      <td class="muted" style="font-size:11px">${fmtTs(p.entered_at||"")}</td>
      ${eodCell(p, cur, sym, mtc)}
      <td style="white-space:nowrap">
        <button class="btn btn-muted btn-sm" style="margin-right:4px"
          onclick="editPos('${mkt}','${s}',${p.qty},${p.entry},${p.stop_loss},${p.target})">Edit</button>
        <button class="btn btn-red btn-sm" onclick="sellPos('${mkt}','${s}')">${btnLabel}</button>
      </td>
    </tr>`;
  }).join("");
}

// ── EOD header badges + per-market banners ────────────────────────────────
function renderEodHeader(d) {
  [["india","NSE"],["us","NYSE"]].forEach(([mkt, label]) => {
    const mtc        = d[mkt+"_mtc"];
    const nPos       = Object.keys(d[mkt].positions||{}).length;
    const harvestMin = d.config.eod_harvest_min || 30;
    const exitMin    = d.config.eod_exit_min    || 15;
    const badge      = document.getElementById(mkt+"-eod");
    const banner     = document.getElementById(mkt+"-eod-banner");
    const phase      = document.getElementById(mkt+"-eod-phase");
    const detail     = document.getElementById(mkt+"-eod-detail");

    if (mtc == null || mtc <= 0) {
      badge.style.display  = "none";
      banner.style.display = "none";
      return;
    }

    // Header badge (visible whenever market is open and within 60 min of close)
    if (mtc <= 60) {
      badge.style.display = "";
      badge.textContent   = `${label} EOD ${Math.floor(mtc)}m`;
      badge.className     = "badge " + (mtc <= exitMin ? "b-eod-exit" : "b-eod-warn");
    } else {
      badge.style.display = "none";
    }

    // Banner above positions table
    if (nPos > 0 && mtc <= harvestMin) {
      banner.style.display = "flex";
      if (mtc <= exitMin) {
        banner.className   = "eod-banner eod-exit";
        phase.textContent  = "⚡ FORCE EXIT";
        phase.style.color  = "var(--red)";
        detail.textContent = `Closing all ${nPos} position${nPos>1?"s":""} — ${Math.floor(mtc)} min until ${label} close`;
      } else {
        banner.className   = "eod-banner eod-warn";
        phase.textContent  = "⏳ EOD HARVEST";
        phase.style.color  = "var(--orange)";
        detail.textContent =
          `Selling profitable positions now. Force exit in ${Math.floor(mtc-exitMin)} min. No new buys.`;
      }
    } else if (nPos > 0 && mtc <= 60) {
      banner.style.display = "flex";
      banner.className     = "eod-banner eod-warn";
      phase.textContent    = "EOD approaching";
      phase.style.color    = "var(--muted)";
      detail.textContent   =
        `${Math.floor(mtc)} min until ${label} close — profit harvest starts in ${Math.floor(mtc-harvestMin)} min`;
    } else {
      banner.style.display = "none";
    }
  });
}

// ── Render signal board ───────────────────────────────────────────────────
function renderSig(analyses, positions, sym, bodyId) {
  const tb = document.getElementById(bodyId);
  if (!analyses||!analyses.length) {
    tb.innerHTML = `<tr><td colspan="11" style="text-align:center;padding:18px;color:var(--muted)">No scan data — start the agent</td></tr>`;
    return;
  }
  tb.innerHTML = analyses.map(a => {
    const inP  = positions&&positions[a.symbol];
    const conf = a.confidence;
    const cc   = conf>65?"var(--green)":conf>40?"var(--yellow)":"var(--muted)";
    const actC = a.score>25?"act-buy":a.score<-25?"act-sell":"act-hold";
    const actT = a.score>25?"BUY":a.score<-25?"SELL":"HOLD";
    const rsiC = a.rsi<35?"var(--green)":a.rsi>65?"var(--red)":"inherit";
    const macd = (a.signals?.MACD?.signal||"").split(" ")[0];
    const bb   = (a.signals?.BB?.signal||"").split(" ")[0];
    const ema  = (a.signals?.EMA?.signal||"").split(" ")[0];
    const vol  = (a.signals?.Vol?.signal||"").split(" ")[0];
    const hw   = a.hist_win_days   ?? 0;
    const ht   = a.hist_total_days ?? 0;
    const hArr = hw > ht/2 ? "↑" : hw < ht/2 ? "↓" : "→";
    const hTxt = ht > 0 ? `${hw}/${ht}${hArr}` : "–";
    const hClr = hw > ht/2 ? "var(--green)" : hw < ht/2 ? "var(--red)" : "var(--muted)";
    const nc   = a.news_count ?? 0;
    const ns   = a.news_score ?? 0;
    const nArr = ns > 0 ? "↑" : ns < 0 ? "↓" : "→";
    const nTxt = nc > 0 ? `${nc}${nArr}` : "–";
    const nClr = ns > 0 ? "var(--green)" : ns < 0 ? "var(--red)" : "var(--muted)";
    return `<tr>
      <td class="mono" style="font-weight:600${inP?";color:var(--blue)":""}">${disp(a.symbol)}${inP?" *":""}</td>
      <td class="mono">${sym}${(a.price||0).toFixed(2)}</td>
      <td class="mono" style="color:${rsiC}">${(a.rsi||0).toFixed(0)}</td>
      <td class="muted">${macd}</td>
      <td class="muted">${bb}</td>
      <td class="muted">${ema}</td>
      <td class="muted">${vol}</td>
      <td class="mono" style="color:${hClr};font-weight:600">${hTxt}</td>
      <td class="mono" style="color:${nClr};font-weight:600">${nTxt}</td>
      <td><div class="cbar"><div class="cbar-bg"><div class="cbar-fg" style="width:${conf}%;background:${cc}"></div></div>
          <span class="mono" style="color:${cc};font-size:11px">${conf.toFixed(0)}%</span></div></td>
      <td><span class="${actC}">${actT}</span></td>
    </tr>`;
  }).join("");
}

// ── Render trade log ──────────────────────────────────────────────────────
function renderLog(trades, id, n=20) {
  const el = document.getElementById(id);
  if (!trades||!trades.length) {
    el.innerHTML = `<div class="log-row"><span class="lt">—</span><span class="muted">No trades yet</span></div>`;
    return;
  }
  el.innerHTML = [...trades].reverse().slice(0,n).map(t => {
    const c = t.kind==="BUY"?"lb":t.kind==="SELL"?"ls":"ll";
    return `<div class="log-row"><span class="lt">${fmtTs(t.time)}</span><span class="${c}">${t.message}</span></div>`;
  }).join("");
}

function renderAllLogs(il, ul) {
  const all = [...(il||[]).map(t=>({...t,mkt:"NSE"})),
               ...(ul||[]).map(t=>({...t,mkt:"NYSE"}))]
    .sort((a,b)=>b.time.localeCompare(a.time)).slice(0,150);
  const el = document.getElementById("all-log");
  if (!all.length) { el.innerHTML=`<div class="log-row"><span class="lt">—</span><span class="muted">No trades yet</span></div>`; return; }
  el.innerHTML = all.map(t=>{
    const c = t.kind==="BUY"?"lb":t.kind==="SELL"?"ls":"ll";
    return `<div class="log-row"><span class="lt">${fmtTs(t.time)}</span>
      <span class="muted" style="min-width:44px;font-size:10px">${t.mkt}</span>
      <span class="${c}">${t.message}</span></div>`;
  }).join("");
}

function renderAgentLog(entries) {
  if (entries) _lastLogEntries = entries;
  const all = _lastLogEntries || [];
  const el  = document.getElementById("agent-log");
  const cnt = document.getElementById("alog-count");
  if (!all.length) {
    el.innerHTML = `<div class="alog-row"><span class="al-ts">—</span><span class="al-lvl al-debug">—</span><span class="muted">No agent activity yet — start the agent</span></div>`;
    if (cnt) cnt.textContent = "0 entries";
    return;
  }
  const filtered = _logFilter === "ALL" ? all
    : all.filter(e => (e.level||"").toUpperCase() === _logFilter);
  if (!filtered.length) {
    el.innerHTML = `<div class="alog-row"><span class="al-ts">—</span><span class="al-lvl al-debug">—</span><span class="muted">No ${_logFilter} entries</span></div>`;
    if (cnt) cnt.textContent = "0 / " + all.length + " entries";
    return;
  }
  el.innerHTML = [...filtered].reverse().map(e => {
    const lc = "al-"+(e.level||"info").toLowerCase();
    return `<div class="alog-row">
      <span class="al-ts">${fmtTs(e.ts||"")}</span>
      <span class="al-lvl ${lc}">${e.level||""}</span>
      <span style="color:var(--text);flex:1">${e.msg||""}</span>
    </div>`;
  }).join("");
  if (cnt) cnt.textContent = filtered.length + (filtered.length < all.length ? " / "+all.length : "") + " entries";
  if (_autoScroll) el.scrollTop = 0;
}

// ── Render agent badge + controls ─────────────────────────────────────────
function renderAgent(a) {
  const ab  = document.getElementById("agent-badge");
  const bt  = document.getElementById("btn-toggle");
  const bp  = document.getElementById("btn-pause");
  if (a.running && !a.paused) {
    ab.className  = "badge b-run"; ab.textContent = "● "+(a.status||"RUNNING").toUpperCase();
    bt.className  = "btn btn-red"; bt.textContent = "■ Stop Agent";
    bp.textContent = "⏸ Pause";
  } else if (a.paused) {
    ab.className  = "badge b-pause"; ab.textContent = "⏸ PAUSED";
    bt.className  = "btn btn-red";   bt.textContent = "■ Stop Agent";
    bp.textContent = "▶ Resume";
  } else {
    ab.className  = "badge b-idle"; ab.textContent = "IDLE";
    bt.className  = "btn btn-green"; bt.textContent = "▶ Start Agent";
    bp.textContent = "⏸ Pause";
  }
  if (a.last_update) document.getElementById("last-upd").textContent = "Last update: "+fmtDt(a.last_update);
}

// ── Load config into form ─────────────────────────────────────────────────
function toggleSettings(el) {
  const on = el.checked;
  const lbl = document.getElementById('settings-toggle-label');
  lbl.textContent = on ? '● ACTIVE' : '○ FULL LIBERTY';
  lbl.style.color = on ? '#3fb950' : '#484f58';
  document.querySelectorAll('.settings-input').forEach(inp => inp.disabled = !on);
  // Turning this off bypasses the confidence gate, position cap, ADX and index
  // filters and the ATR SL/TP clamps, so the server requires explicit
  // confirmation. Reverting to ON needs no confirmation.
  const body = on
    ? {settings_enabled: true}
    : {settings_enabled: false, confirm_disable_settings: true};
  if (!on && !confirm('FULL LIBERTY mode bypasses the confidence threshold, position cap, '
                    + 'ADX and index filters and the ATR stop clamps.\n\nDisable risk controls?')) {
    el.checked = true;
    lbl.textContent = '● ACTIVE';
    lbl.style.color = '#3fb950';
    document.querySelectorAll('.settings-input').forEach(inp => inp.disabled = false);
    return;
  }
  fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  }).then(r => r.json()).then(d => {
    if (d && d.ok === false) toast('Rejected: ' + (d.msg || 'invalid settings'));
  }).catch(() => toast('Could not update settings'));
}

function loadCfg(c) {
  document.getElementById("s-risk").value  = c.risk_per_trade;
  document.getElementById("s-conf").value  = c.confidence_threshold;
  document.getElementById("s-sl").value    = c.stop_loss_pct;
  document.getElementById("s-tgt").value   = c.target_pct;
  document.getElementById("s-chk").value   = c.check_interval_min;
  document.getElementById("s-idle").value  = c.idle_interval_min;
  document.getElementById("s-ip").value    = c.india_max_positions;
  document.getElementById("s-up").value    = c.us_max_positions;
  document.getElementById("s-eod-h").value = c.eod_harvest_min;
  document.getElementById("s-eod-e").value = c.eod_exit_min;
  document.getElementById("s-rl-exit").value = c.rl_exit_confidence ?? 55;
  // Sync settings toggle state
  const stToggle = document.getElementById('settings-toggle');
  if (stToggle && c.settings_enabled !== undefined) {
    const on = !!c.settings_enabled;
    stToggle.checked = on;
    const lbl = document.getElementById('settings-toggle-label');
    if (lbl) { lbl.textContent = on ? '● ACTIVE' : '○ FULL LIBERTY'; lbl.style.color = on ? '#3fb950' : '#484f58'; }
    document.querySelectorAll('.settings-input').forEach(inp => inp.disabled = !on);
  }
}

// ── Load editable state into form ─────────────────────────────────────────
function loadEditState(d) {
  const fill = (id, val, decimals=2) =>
    { const el = document.getElementById(id); if(el) el.value = (val||0).toFixed ? (val||0).toFixed(decimals) : (val||0); };
  fill("es-india-cash",   d.india.cash);
  fill("es-india-rpnl",   d.india.realised_pnl);
  fill("es-india-wins",   d.india.wins,    0);
  fill("es-india-losses", d.india.losses,  0);
  fill("es-india-peak",   d.india.peak_portfolio || d.india.portfolio_value);
  fill("es-us-cash",      d.us.cash);
  fill("es-us-rpnl",      d.us.realised_pnl);
  fill("es-us-wins",      d.us.wins,    0);
  fill("es-us-losses",    d.us.losses,  0);
  fill("es-us-peak",      d.us.peak_portfolio || d.us.portfolio_value);
}

async function saveEditState() {
  const gf = id => parseFloat(document.getElementById(id).value);
  const gi = id => parseInt(document.getElementById(id).value);
  const body = {
    india: { cash: gf("es-india-cash"), realised_pnl: gf("es-india-rpnl"),
             wins: gi("es-india-wins"), losses: gi("es-india-losses"),
             peak_portfolio: gf("es-india-peak") },
    us:    { cash: gf("es-us-cash"),    realised_pnl: gf("es-us-rpnl"),
             wins: gi("es-us-wins"),    losses: gi("es-us-losses"),
             peak_portfolio: gf("es-us-peak") },
  };
  const r = await fetch("/api/edit/state", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  const d = await r.json();
  toast(d.ok ? "Portfolio state updated" : "Update failed", d.ok);
  if (d.ok) setTimeout(doRefresh, 300);
}

// ── Position editing ──────────────────────────────────────────────────────
function editPos(mkt, sym, qty, entry, sl, tgt) {
  document.getElementById("epm-sym").textContent = disp(sym);
  document.getElementById("epm-mkt").value = mkt;
  document.getElementById("epm-sym-h").value = sym;
  document.getElementById("epm-qty").value   = qty;
  document.getElementById("epm-entry").value = entry.toFixed(2);
  document.getElementById("epm-sl").value    = sl.toFixed(2);
  document.getElementById("epm-tgt").value   = tgt.toFixed(2);
  document.getElementById("pos-edit-modal").classList.add("open");
}

function closeEditPos() {
  document.getElementById("pos-edit-modal").classList.remove("open");
}

async function saveEditPos() {
  const mkt = document.getElementById("epm-mkt").value;
  const sym = document.getElementById("epm-sym-h").value;
  const body = {
    qty:       parseInt(document.getElementById("epm-qty").value),
    entry:     parseFloat(document.getElementById("epm-entry").value),
    stop_loss: parseFloat(document.getElementById("epm-sl").value),
    target:    parseFloat(document.getElementById("epm-tgt").value),
  };
  const r = await fetch(`/api/edit/position/${mkt}/${sym}`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  const d = await r.json();
  toast(d.ok ? "Position updated" : d.msg, d.ok);
  closeEditPos();
  if (d.ok) setTimeout(doRefresh, 300);
}

// ── Sessions ──────────────────────────────────────────────────────────────
async function pollSessions() {
  try {
    const r = await fetch("/api/sessions");
    if (!r.ok) return;
    _sessData = await r.json();
    renderSessions(_sessData);
  } catch(e) {}
}

function setSessFilter(f, el) {
  _sessFilter = f;
  document.querySelectorAll(".sess-filter").forEach(b => b.classList.remove("on"));
  el.classList.add("on");
  renderSessions(_sessData);
}

function toggleSessBody(safeId) {
  const el = document.getElementById("sb-"+safeId);
  if (el) el.classList.toggle("open");
}

function renderSessions(sessions) {
  const filtered = _sessFilter === "all" ? sessions
    : sessions.filter(s => s.market === _sessFilter);
  const cnt = document.getElementById("sess-count");
  if (cnt) cnt.textContent = filtered.length + " session" + (filtered.length !== 1 ? "s" : "");
  const tc = document.getElementById("sess-tab-cnt");
  if (tc) tc.textContent = sessions.length > 0 ? "("+sessions.length+")" : "";
  const el = document.getElementById("sess-list");
  if (!filtered.length) {
    el.innerHTML = `<div style="text-align:center;padding:36px;color:var(--muted)">No past sessions yet — sessions are archived automatically when the calendar date changes</div>`;
    return;
  }
  el.innerHTML = filtered.map(s => {
    const sm      = s.market === "india" ? "₹" : "$";
    const safe    = (s.id||"").replace(/[^a-zA-Z0-9]/g,"_");
    const pnlCls  = (s.net_pnl||0) >= 0 ? "green" : "red";
    const tot     = (s.wins||0) + (s.losses||0);
    const wr      = tot > 0 ? (s.wins/tot*100).toFixed(0)+"%" : "—";
    const mktCls  = s.market === "india" ? "sess-mkt-india" : "sess-mkt-us";
    const mktLbl  = s.market === "india" ? "NSE" : "NYSE";
    const sign    = (s.net_pnl||0) >= 0 ? "+" : "";
    const trades  = s.trades || [];
    const trHtml  = trades.length
      ? [...trades].reverse().map(t => {
          const c = t.kind==="BUY"?"lb":t.kind==="SELL"?"ls":"ll";
          return `<div class="log-row"><span class="lt">${fmtTs(t.time)}</span><span class="${c}">${t.message}</span></div>`;
        }).join("")
      : `<div class="log-row"><span class="muted" style="padding:4px 10px">No trades this session</span></div>`;
    return `<div class="sess-card">
      <div class="sess-hdr" onclick="toggleSessBody('${safe}')">
        <span class="sess-date">${s.date||"—"}</span>
        <span class="sess-mkt ${mktCls}">${mktLbl}</span>
        <div style="flex:1;display:flex;flex-wrap:wrap;gap:16px;align-items:center;padding:0 6px">
          <span class="mono" style="font-size:12px;color:var(--muted)">
            ${sm}${(s.start_cash||0).toLocaleString(undefined,{maximumFractionDigits:0})}
            → ${sm}${(s.end_cash||0).toLocaleString(undefined,{maximumFractionDigits:0})}
          </span>
          <span class="mono ${pnlCls}" style="font-size:14px;font-weight:700">
            ${sign}${sm}${Math.abs(s.net_pnl||0).toLocaleString(undefined,{maximumFractionDigits:0})}
            <span style="font-size:11px;font-weight:400;color:inherit;opacity:.8">
              (${sign}${(s.pnl_pct||0).toFixed(1)}%)
            </span>
          </span>
          <span class="muted" style="font-size:11px">
            ${s.wins||0}W ${s.losses||0}L · ${wr} WR · ${s.n_trades||trades.length} trade${(s.n_trades||trades.length)!==1?"s":""}
          </span>
        </div>
        <span style="color:var(--muted);font-size:14px;padding-left:4px">›</span>
      </div>
      <div class="sess-body" id="sb-${safe}">
        <div class="sess-metrics">
          <div class="sess-m">
            <span class="sess-m-l">Start Capital</span>
            <span class="sess-m-v">${sm}${(s.start_cash||0).toLocaleString(undefined,{maximumFractionDigits:0})}</span>
          </div>
          <div class="sess-m">
            <span class="sess-m-l">End Cash</span>
            <span class="sess-m-v">${sm}${(s.end_cash||0).toLocaleString(undefined,{maximumFractionDigits:0})}</span>
          </div>
          <div class="sess-m">
            <span class="sess-m-l">End Portfolio</span>
            <span class="sess-m-v">${sm}${(s.end_portfolio||s.end_cash||0).toLocaleString(undefined,{maximumFractionDigits:0})}</span>
          </div>
          <div class="sess-m">
            <span class="sess-m-l">Net P&amp;L</span>
            <span class="sess-m-v ${pnlCls}">${sign}${sm}${Math.abs(s.net_pnl||0).toLocaleString(undefined,{maximumFractionDigits:0})}</span>
          </div>
          <div class="sess-m">
            <span class="sess-m-l">Wins / Losses</span>
            <span class="sess-m-v">${s.wins||0}W / ${s.losses||0}L</span>
          </div>
          <div class="sess-m">
            <span class="sess-m-l">Win Rate</span>
            <span class="sess-m-v">${wr}</span>
          </div>
          <div class="sess-m">
            <span class="sess-m-l">Archived at</span>
            <span class="sess-m-v muted" style="font-size:11px">${fmtTs(s.archived_at||"")}</span>
          </div>
        </div>
        <div class="log-box" style="max-height:200px;border-radius:0;border:none;background:#0a0e14">
          ${trHtml}
        </div>
      </div>
    </div>`;
  }).join("");
}

async function closeSessionNow(market) {
  if (!confirm(`Archive current ${market.toUpperCase()} session and reset to starting capital?\n\nAll open positions will be force-closed at current prices.`)) return;
  const r = await fetch("/api/sessions/close/"+market, {method:"POST"});
  const d = await r.json();
  toast(d.msg, d.ok);
  if (d.ok) { setTimeout(doRefresh, 400); setTimeout(pollSessions, 600); }
}

// ── Main refresh ──────────────────────────────────────────────────────────
async function doRefresh() {
  try {
    const r = await fetch("/api/state");
    const d = await r.json();
    _cfg = d.config;
    _lastState = d;
    renderStats(d);
    renderAgent(d.agent);
    loadCfg(d.config);
    loadEditState(d);
    renderEodHeader(d);
    renderPos(d.india.positions, d.signals.india_prices, "₹", "india-pos", "india-pc", "india", d.india_mtc);
    renderPos(d.us.positions,    d.signals.us_prices,    "$",  "us-pos",    "us-pc",    "us",    d.us_mtc);
    renderSig(d.signals.india, d.india.positions, "₹", "india-sig");
    renderSig(d.signals.us,    d.us.positions,    "$",  "us-sig");
    renderLog(d.india.trade_log, "india-log");
    renderLog(d.us.trade_log,    "us-log");
    renderAllLogs(d.india.trade_log, d.us.trade_log);
    renderAgentLog(d.agent_log || []);
    if (d.decision_log) { _lastDlogEntries = d.decision_log; renderDlog(d.decision_log); }
    // update sessions tab badge + refresh sessions data in background
    const tc = document.getElementById("sess-tab-cnt");
    if (tc) tc.textContent = d.sessions_count > 0 ? "("+d.sessions_count+")" : "";
    if (_activeTab === "sessions") pollSessions();
    startProg();
  } catch(e) { toast("Refresh failed: "+e.message, false); }
}

// ── SSE live price updates ─────────────────────────────────────────────────
(function startPriceSSE() {
  const src = new EventSource('/api/prices/stream');
  src.onmessage = (e) => {
    try {
      const prices = JSON.parse(e.data);
      if (!_lastState) return;
      // recalculate PnL for each market using cached positions + fresh prices
      ['india', 'us'].forEach(mkt => {
        const sym  = mkt === 'india' ? '₹' : '$';
        const m    = _lastState[mkt];
        const cap  = mkt === 'india' ? _cfg.india_capital : _cfg.us_capital;
        if (!m) return;
        // merge: latest prices override signal prices
        const merged = Object.assign({}, _lastState.signals[mkt+'_prices'], prices);
        let upnl = 0;
        Object.entries(m.positions || {}).forEach(([s, p]) => {
          const cur = merged[s] || p.entry;
          upnl += p.side === 'short'
            ? (p.entry - cur) * p.qty
            : (cur - p.entry) * p.qty;
        });
        const pv    = m.cash + Object.entries(m.positions||{}).reduce((acc,[s,p])=>{
          const cur = merged[s]||p.entry;
          return acc + (p.side==='short' ? (p.entry-cur)*p.qty : cur*p.qty);
        }, 0);
        const tpnl  = (m.realised_pnl||0) + upnl;
        const df    = pv - cap, pct = cap > 0 ? df/cap*100 : 0;
        const fc    = (v,s) => s+(Math.abs(v)<1e6 ? Math.abs(v).toLocaleString(undefined,{maximumFractionDigits:0}) : (Math.abs(v)/1e5).toFixed(1)+'L');
        const sc    = v => v>=0?'green':'red';
        const sgn   = v => v>=0?'+':'-';
        const el    = id => document.getElementById(mkt+'-'+id);
        if (el('pv'))   { el('pv').textContent = fc(pv,sym); el('pv').className='c-val mono '+sc(df); }
        if (el('diff')) el('diff').innerHTML = `<span class="${sc(df)}">${sgn(df)}${fc(df,sym)} (${sgn(pct)}${Math.abs(pct).toFixed(1)}%)</span>`;
        if (el('tpnl')) { el('tpnl').textContent=(tpnl>=0?'+':'')+fc(tpnl,sym); el('tpnl').className='c-val mono '+sc(tpnl); }
        if (el('upnl')) el('upnl').innerHTML=`Float: <span class="${sc(upnl)}">${sgn(upnl)}${fc(upnl,sym)}</span>`;
      });
    } catch(_) {}
  };
  src.onerror = () => { src.close(); setTimeout(startPriceSSE, 5000); };
})();

function scheduleRefresh() {
  clearTimeout(_refreshTimer);
  _refreshTimer = setTimeout(()=>{ doRefresh(); scheduleRefresh(); }, INTERVAL*1000);
}

// ── Controls ──────────────────────────────────────────────────────────────
async function toggleAgent() {
  const running = document.getElementById("btn-toggle").textContent.includes("Stop");
  const url = running ? "/api/agent/stop" : "/api/agent/start";
  const r = await fetch(url,{method:"POST"});
  const d = await r.json();
  toast(d.msg, d.ok);
  setTimeout(doRefresh, 400);
}

async function pauseAgent() {
  const r = await fetch("/api/agent/pause",{method:"POST"});
  const d = await r.json();
  if (d.ok) toast(d.paused ? "Agent paused" : "Agent resumed");
  else toast(d.msg, false);
  setTimeout(doRefresh, 300);
}

async function sellPos(market, symbol) {
  if (!confirm("Force sell "+disp(symbol)+"?")) return;
  const r = await fetch("/api/sell/"+market+"/"+symbol,{method:"POST"});
  const d = await r.json();
  toast(d.msg, d.ok);
  setTimeout(doRefresh, 400);
}

async function saveConfig() {
  const p = {
    risk_per_trade:       parseFloat(document.getElementById("s-risk").value),
    confidence_threshold: parseInt(document.getElementById("s-conf").value),
    stop_loss_pct:        parseFloat(document.getElementById("s-sl").value),
    target_pct:           parseFloat(document.getElementById("s-tgt").value),
    check_interval_min:   parseInt(document.getElementById("s-chk").value),
    idle_interval_min:    parseInt(document.getElementById("s-idle").value),
    india_max_positions:  parseInt(document.getElementById("s-ip").value),
    us_max_positions:     parseInt(document.getElementById("s-up").value),
    eod_harvest_min:      parseInt(document.getElementById("s-eod-h").value),
    eod_exit_min:         parseInt(document.getElementById("s-eod-e").value),
    rl_exit_confidence:   parseInt(document.getElementById("s-rl-exit").value),
  };
  const r = await fetch("/api/config",{method:"POST",
    headers:{"Content-Type":"application/json"},body:JSON.stringify(p)});
  const d = await r.json();
  toast(d.ok ? "Settings saved — applies next cycle" : ("Save rejected: " + (d.msg || "invalid value")), d.ok);
}

async function resetMkt(market) {
  if (!confirm("Reset "+market+" paper state? This erases all positions, trades and P&L.")) return;
  const r = await fetch("/api/reset/"+market,{method:"POST"});
  const d = await r.json();
  toast(d.msg, d.ok);
  setTimeout(doRefresh, 300);
}

// ── Boot ──────────────────────────────────────────────────────────────────
doRefresh();
scheduleRefresh();
startProg();
