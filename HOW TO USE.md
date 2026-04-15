# Apex Trading Dashboard — How To Use
### A complete guide for absolute beginners

---

## What is this?

Apex is a **paper trading bot** with a web dashboard you can open in your browser.

**Paper trading = fake money.** The bot watches real stock prices and makes real buy/sell decisions — but no actual money ever moves. It's a safe way to test a trading strategy.

- **Morning (Indian market hours):** Watches 16 top NIFTY 50 stocks on NSE
- **Night (US market hours):** Watches 16 top tech stocks on NYSE/NASDAQ
- **Your screen:** A live dashboard at `http://localhost:7000` showing everything

---

## Before You Start (One-Time Setup)

You only need to do this once, ever.

### Step 1 — Install Python

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Click the big yellow **Download Python** button
3. Run the installer
4. **Important:** On the first screen of the installer, tick the box that says **"Add Python to PATH"** before clicking Install

### Step 2 — Open the Apex folder in VSCode

1. Open **VSCode**
2. Click **File → Open Folder**
3. Navigate to `c:\Anubhav\C Programming\Apex` and click **Select Folder**

### Step 3 — Open the Terminal inside VSCode

Press **Ctrl + `** (the backtick key, top-left of your keyboard, same key as `~`)

A panel will appear at the bottom of VSCode. This is your terminal — you type commands here.

### Step 4 — Install the required packages

Click inside the terminal panel, paste this line, and press **Enter**:

```
pip install flask yfinance pandas numpy
```

Wait for it to finish. You'll see a lot of text scrolling — that's normal. When the blinking cursor comes back, it's done.

---

## Running the Dashboard

Every time you want to use Apex:

1. Open **VSCode** with the Apex folder open
2. Press **Ctrl + `** to open the terminal
3. Type this and press **Enter**:
   ```
   python apex_dashboard.py
   ```
4. Your browser will automatically open to the dashboard after a couple of seconds

To **stop** the bot, click inside the terminal and press **Ctrl + C**.

> **Tip:** Don't close VSCode while the bot is running — that will stop it.

---

## The Dashboard — What Everything Means

### Top Bar

| Thing | What it means |
|---|---|
| **NSE OPEN / CLOSED** | Whether the Indian stock market is currently trading |
| **NYSE OPEN / CLOSED** | Whether the US stock market is currently trading |
| **The coloured dot badge** | The bot's current status (IDLE / RUNNING / PAUSED) |
| **▶ Start Agent** button | Turns the bot on |
| **■ Stop Agent** button | Turns the bot off (appears when running) |
| **⏸ Pause** button | Temporarily freezes the bot without stopping it |

---

### Summary Cards (the 6 boxes per market)

Each market (India and US) has 6 information boxes:

| Box | What it shows |
|---|---|
| **Portfolio** | Total value of your virtual wallet right now (cash + any open stock positions) |
| **Cash** | How much fake money is sitting uninvested |
| **Realised P&L** | Profit or loss on trades you have **already closed** (locked in) |
| **Win Rate** | Percentage of closed trades that made money |
| **Drawdown** | How far your portfolio has dropped from its highest point — lower is better |
| **Total Profit** | Realised profit + the floating profit on positions you still hold. The small line below ("Float") shows just the open-position profit |

Green numbers = profit. Red numbers = loss.

---

### Tabs

#### India (NSE) / US (NYSE) tabs

- **Open Positions** — stocks the bot currently holds
  - Shows entry price, current price, live P&L per position, stop-loss level, and target price
  - The **Sell** button lets you manually close any position immediately

- **Signal Board** — the bot's live analysis of every stock it watches
  - RSI, MACD, Bollinger Bands, EMA, Volume are technical indicators — the bot uses them to score each stock
  - The **Confidence** bar shows how strongly the bot feels about a trade (higher = stronger signal)
  - **BUY / HOLD / SELL** is the bot's current recommendation for that stock

- **Recent Trades** — a history of the last few buys and sells in that market

#### Settings tab

Change how the bot behaves without restarting it. Settings apply from the next cycle onward.

| Setting | What it does | Default |
|---|---|---|
| **Risk per Trade** | Fraction of available cash used per buy (0.12 = 12%) | 12% |
| **Confidence Threshold** | Minimum score before the bot will buy anything | 62% |
| **Stop Loss** | How far a stock can drop before the bot auto-sells to cut losses (0.03 = 3%) | 3% |
| **Target** | How much profit triggers an auto-sell (0.045 = 4.5%) | 4.5% |
| **Check Interval** | How often the bot scans stocks when a market is open (minutes) | 3 min |
| **Idle Interval** | How often the bot checks when both markets are closed | 15 min |
| **India / US Max Positions** | Maximum number of stocks held at once per market | 4 each |

Click **Save Settings** after making changes.

> **Danger Zone** — The red Reset buttons wipe all trade history and return the wallet to its starting balance. Use only if you want a completely fresh start.

#### All Logs tab

Two sections:

- **Agent System Log** — a real-time feed of everything the bot is doing behind the scenes (scanning, saving, waiting). Also saved permanently to `apex.log` which you can open directly in VSCode.
- **All Trade Activity** — every buy and sell from both markets in one combined list, newest first

---

## The Files in Your Folder

| File | What it is |
|---|---|
| `apex_dashboard.py` | The main program — run this |
| `apex_dual_state.json` | Your portfolio's saved state (created automatically on first run) |
| `apex.log` | A permanent text log of everything the bot has ever done — open it in VSCode anytime |
| `HOW TO USE.md` | This guide |

---

## Frequently Asked Questions

**Q: The browser didn't open automatically.**
Open your browser manually and go to: `http://localhost:7000`

**Q: I see "NSE CLOSED" and "NYSE CLOSED" and the bot isn't doing anything.**
That's correct — both markets are currently outside trading hours. The bot will automatically start scanning when a market opens. It still runs in the background checking every 15 minutes.

**India market hours (IST):** Monday–Friday, 9:15 AM – 3:30 PM
**US market hours (IST):** Monday–Friday, approximately 7:00 PM – 1:30 AM (next day)

**Q: Can I lose real money?**
No. This is paper trading only. No real brokerage account is connected. No real money moves.

**Q: I closed VSCode by accident.**
The bot stopped. Reopen VSCode, press **Ctrl + `**, and run `python apex_dashboard.py` again. Your portfolio data is saved in `apex_dual_state.json` so nothing is lost.

**Q: The dashboard shows wrong data or freezes.**
Click the **↻ Refresh** button in the top right corner of the dashboard.

**Q: How do I reset everything and start fresh?**
Go to **Settings** tab → scroll to the bottom → click **Reset Everything**.

**Q: How do I see the full log history?**
In VSCode, click on `apex.log` in the file list on the left. It contains every action the bot has ever taken, even from previous sessions.

---

## Quick Reference

| Action | How |
|---|---|
| Open terminal in VSCode | **Ctrl + `** |
| Start the bot | In terminal: `python apex_dashboard.py` |
| Open the dashboard | Browser: `http://localhost:7000` |
| Stop the bot | Click terminal → **Ctrl + C** |
| View full logs | Open `apex.log` in VSCode |
