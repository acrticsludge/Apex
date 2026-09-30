# Your first deploy - step by step

This is written for someone who has never used this codebase before. Every step
says what to type and what it means. Nothing here is optional except where it
says so.

**Time needed:** about 30 minutes, most of it waiting for builds.

**What you are doing:** putting this bot online so it can trade on live market
data (with fake money - it never places real orders).

---

## Before you start: what you need

You need accounts at these places. Nothing is public, all have free tiers.

| What | Why | Where |
|---|---|---|
| A GitHub account | Holds the code | github.com |
| A Railway account | Runs the server | railway.app |
| A Supabase project | Stores your trade history | supabase.com |

You already have a Supabase project - it is configured in your `.env`.

---

## Step 1 - check your setup

Open PowerShell in the project folder and run:

```powershell
cd "C:\Users\Anish\Downloads\Apex (2)_actualagent\Apex"
.\.venv\Scripts\python.exe preflight.py
```

**What it does:** reads your `.env` file and checks for problems that would break
the deploy or lose data.

**What you want to see:** `Safe to deploy` at the bottom.

Right now it will also show three `??` warnings. Those are fine - they mean
"optional improvement", not "broken". We deal with two of them in steps 5 and 6.

If you see `!!` instead, **stop and fix it**. The message tells you what to do.

---

## Step 2 - push your code to GitHub

Your code is already committed and pushed. To confirm:

```powershell
git log --oneline -3
git status
```

**What you want to see:** three lines with short commit messages, and
`working tree clean` (or nothing after the branch name).

If it says something else, tell me and we'll sort it.

---

## Step 3 - create the pull request

A pull request means "please review this before it goes live". You have changed
a lot, so this step matters.

1. Open this link in your browser:
   `https://github.com/acrticsludge/Apex/compare/main...feat/jev-integration?expand=1`

2. GitHub shows every file you changed.
3. Click the green **"Create pull request"** button.
4. Add a title: `Harden trading agent: validation gate, slippage, alerting`
5. Click **"Create pull request"**.

**What happens next:** GitHub starts running your tests automatically. Wait about
3 minutes, then refresh the page.

**What you want to see:** a green box saying "All checks have passed".

If any check is red, do not merge. Send me the red box.

---

## Step 4 - deploy to Railway

Railway turns your GitHub code into a running server.

1. Open `railway.app` and log in.
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Find your Apex repo and select it.
4. Railway builds and deploys automatically. This takes **5-10 minutes** the
   first time (it installs a lot of software).

### Step 4a - set the login password

**This is the most important step. Skip it and the site will be unreachable.**

1. In Railway, click your project -> click the service -> the **Variables** tab.
2. Click **+ New** and add these three, one at a time:

   - **Name:** `APEX_PASS`
     **Value:** a strong password you invent now. Write it down somewhere safe.
     Make it at least 16 characters.

   - **Name:** `APEX_USER`
     **Value:** anything except the word `apex`. For example: `mytrader`

   - **Name:** `APEX_SECRET`
     **Value:** open PowerShell in a new window and run
     `python -c "import secrets; print(secrets.token_hex(32))"`
     then paste what it prints.

3. Click **Deploy / Redeploy** if Railway offers it.

**What each one does:**

| Variable | What happens if you skip it |
|---|---|
| `APEX_PASS` | **The site refuses all logins.** You get a "Server credentials are not configured" error and cannot get in. |
| `APEX_USER` | Same - the app explicitly refuses the default username `apex`. |
| `APEX_SECRET` | The site works, but you get logged out whenever the server restarts or runs more than one process. |

---

## Step 5 - get told if it stops working (10 minutes)

Right now, if the bot crashed at 3am, nobody would know until you happened to
look. Let's fix that.

### Option A - get an email when the site goes down (easiest)

1. Open `uptimerobot.com` and create a free account.
2. Click **Add New Monitor**.
3. Fill in:
   - **Monitor Type:** HTTP(s)
   - **Friendly Name:** Apex Trading Bot
   - **URL (or IP):** the Railway URL, which looks like
     `https://your-app-name.up.railway.app/healthz` - paste your real one and add
     `/healthz` at the end
4. **Monitoring Interval:** 5 minutes
5. Click **Create Monitor**.

**What happens:** you get an email if the bot stops responding. It checks every
5 minutes, forever, for free.

### Option B - get a message when the bot stops trading (better)

1. Open `ntfy.sh` - no account needed, it's free.
2. On your phone, install the ntfy app.
3. In the app, subscribe to a topic called `apex-alerts`.
4. In Railway's **Variables** tab, add:
   - **Name:** `APEX_ALERT_WEBHOOK_URL`
   - **Value:** `https://ntfy.sh/apex-alerts`
5. Redeploy.

**What happens:** your phone buzzes when the bot stops trading, when it recovers,
and when the trading loop dies unexpectedly.

**Why this matters most:** if the trading thread crashes, nothing appears on the
website at all. The site looks perfectly fine while the bot is dead. This is the
only way to find out.

---

## Step 6 - make the bot's learning stick (5 minutes)

This one is optional but recommended.

Right now, every time Railway deploys, the bot's learned trading knowledge is
thrown away and it starts again from the model built into the code. This fixes
that.

1. In Railway, open your project's **Settings** - the gear icon.
2. Find the service, scroll to **Volumes**, click **+ New Volume**.
3. **Mount Path:** `/data`
4. **Mount Path:** `/data` - name the volume anything, like `apex-data`.
5. Save.
6. Back in **Variables**, add:
   - **Name:** `TRADING_AGENT_STORAGE_DIR`
     **Value:** `/data`
7. Redeploy.

**What happens:** the bot's learning now survives redeploys.

---

## Step 7 - try it before letting it trade

**Do not skip this.** First run on real market data is where surprises happen.

### Step 7a - watch it start

1. Open your Railway URL in a browser.
2. Log in with the username and password from step 4a.
3. You should see two markets: **India** and **US**.

### Step 7b - check whether the markets are open

The bot **only trades during market hours**:

| Market | Trading hours (India time) | In US Eastern |
|---|---|---|
| India (NSE) | 9:15am - 3:30pm | 9:30am - 12:00am |
| US (NYSE/Nasdaq) | 7:00pm - 2:00am | 9:30am - 4:00pm |

If both are closed, the bot correctly does nothing. That is not a bug.

### Step 7c - read what it's thinking

On the dashboard, the **Think Log** panel shows every decision in plain English,
like:

```
SCAN  RELIANCE.NS  RSI=54.2  Score=+35  ^long(71%)  ADX=28.4
ENTRY  RELIANCE.NS @ 2450.50  SL:2410.00  T:2530.00
```

This is the bot explaining itself. Read it. If you cannot follow why it bought
something, do not let it trade.

---

## Step 8 - when something goes wrong

### The site says "Server credentials are not configured"

You skipped step 4a. Add `APEX_PASS` and `APEX_USER` in Railway, then redeploy.

### You cannot log in at all

Your username is probably still `apex`, which the app refuses on purpose. Check
Railway -> Variables.

### The site loads but says "No trades"

Almost certainly the market is closed. Check the hours in step 7b.

### You get "Service has failed to start" in Railway

Click the failed deployment to see the log. The most common cause is missing
`APEX_PASS`. You can also re-run the preflight check locally to compare.

### You want to undo the deploy

Railway -> your service -> **Settings** -> **Rollback** -> pick the earlier
deployment.

**Note:** if you did step 6, rolling back does *not* roll back the bot's learned
trading data - the volume keeps the newer version. That's intentional.

---

## What is still not done

Being straight with you about what I could not fix. Both need information only
you have.

**Indian trading costs are wrong.** Your model charges a flat 0.06% commission
to both markets. That is roughly right for US stocks but too low for Indian ones,
which also pay STT (securities transaction tax), stamp duty, GST and exchange
charges. So your Indian results will look slightly better than reality.

I did not fix this because I would have had to invent the numbers, and they
depend on your broker and whether you trade intraday or hold overnight. Look at
your broker's contract note for the actual charges, then tell me and I'll wire
them in properly. Inventing figures that look authoritative would be worse than
leaving the honest default.

**Your dashboard is on a public internet address.** Anyone who finds the URL can
reach the login page. Your password protects it, but adding an IP allowlist
(e.g. Cloudflare Tunnel) would mean only your home connection can load it.

---

## Quick reference

```powershell
# Start the app on your own computer
.\.venv\Scripts\python.exe apex_dashboard.py
# then open http://localhost:7000

# Check whether a deploy is safe
.\.venv\Scripts\python.exe preflight.py

# Run all tests
.\.venv\Scripts\python.exe -m pytest

# Watch what the bot is thinking
Get-Content apex.log -Wait -Tail 30
```

**If anything in here is confusing or a step fails, tell me which step number
and what you saw. That's all I need to help.**
