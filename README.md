# PulmoScan

A web app that helps doctors screen lung CT scans. A doctor signs up, adds patients, and uploads scans. A ResNet18 + SVM model labels each scan **Benign**, **Malignant** or **Normal**. An AI assistant answers questions about the patient's results.

> **Medical disclaimer:** The model is 93.6% accurate on a held-out test set of 1,478 images. It gives no per-scan confidence. It is a research tool, not a diagnosis. A clinician must confirm every result.

---

## Contents

1. [How it works](#how-it-works)
2. [Repository layout](#repository-layout)
3. [Run locally](#run-locally)
4. [The chatbot: Ollama and alternatives](#the-chatbot-ollama-and-alternatives)
5. [Deploy to Render](#deploy-to-render)
6. [Running costs and day-to-day operations](#running-costs-and-day-to-day-operations)
7. [Other hosting options](#other-hosting-options)
8. [Troubleshooting](#troubleshooting)
9. [Security](#security)

---

## How it works

**Features:** email signup with a 6-digit code (sent through Gmail), Google sign-in, patient records, scan upload with history, and a per-patient AI chat assistant.

**Prediction pipeline** (in `predict_lung_condition()` in `app.py`):

```
CT image → contrast stretch → pad/resize to 512×512 → lung segmentation (KMeans + scikit-image)
         → ResNet18 features (feature_extractor.pth) → StandardScaler (scaler.pkl) → SVM (svm_model.pkl)
         → Benign / Malignant / Normal
```

If an image can't be read or processed, the app says so and saves nothing. It never guesses a result.

**Stack:** Python 3.13, Flask 3.1, Flask-SQLAlchemy (SQLite), Flask-Mail, Flask-Limiter, Authlib, PyTorch 2.14 (CPU only), scikit-learn 1.6, OpenCV, gunicorn.

---

## Repository layout

```
Web_Development_Project/
├── projectFinal/            ← the web app: this is what you run and deploy
│   ├── app.py               ← Flask app: routes, database, model inference, chatbot
│   ├── test_app.py          ← safety tests
│   ├── requirements.txt     ← Python packages
│   ├── render.yaml          ← Render deployment config (Blueprint)
│   ├── .env.example         ← template for your settings (copy to .env)
│   ├── feature_extractor.pth, scaler.pkl, svm_model.pkl   ← trained model (~54 MB)
│   ├── templates/           ← HTML pages
│   ├── static/              ← CSS and images
│   └── instance/            ← created at runtime: database + uploaded scans (gitignored)
└── AI_End/                  ← model training notebook (lung.ipynb) and research paper
```

Only `projectFinal/` is needed to run or deploy the app. `AI_End/` is for retraining the model.

---

## Run locally

### 1. Install the prerequisites

| Tool | Version | Get it |
|---|---|---|
| Python | **3.13** | https://www.python.org/downloads/ (on Windows, tick **"Add Python to PATH"**) |
| Git | any | https://git-scm.com/downloads |
| Ollama | optional, chatbot only | https://ollama.com/download |

Check Python: `python --version` should print `Python 3.13.x`.

### 2. Get the code

```bash
git clone https://github.com/Sifat2128/Web_Development_Project.git
cd Web_Development_Project/projectFinal
```

### 3. Create a virtual environment and install packages (first time only)

A virtual environment keeps this project's packages separate from the rest of your computer.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If PowerShell says *"running scripts is disabled"*, run this once, then activate again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Windows (Git Bash) / macOS / Linux:**
```bash
python -m venv .venv
source .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Your prompt now starts with `(.venv)`. The install takes a few minutes because PyTorch is large. It installs the CPU-only build, which is all the app needs. `gunicorn` is skipped on Windows, where it doesn't run; you don't need it locally.

### 4. Create your settings file

```bash
cp .env.example .env               # PowerShell: Copy-Item .env.example .env
```

Open `.env` and fill it in. **Never commit `.env`.** It is already in `.gitignore`.

| Setting | Required? | What to put |
|---|---|---|
| `SECRET_KEY` | **Yes.** The app refuses to start without it. | A long random string: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `MAIL_USERNAME` | Yes, for signup | The Gmail address that sends verification codes |
| `MAIL_PASSWORD` | Yes, for signup | A Gmail **app password**, *not* your normal password (see below) |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Optional | Only for "Sign in with Google" (see below) |
| `OLLAMA_URL` / `OLLAMA_MODEL` / `OLLAMA_API_KEY` | Optional | Only for the chatbot (see [the chatbot section](#the-chatbot-ollama-and-alternatives)) |
| `DATA_DIR` | No (local) | Leave unset locally. Defaults to `projectFinal/instance/`. |

#### Get a Gmail app password

1. Sign in to the Gmail account at https://myaccount.google.com.
2. Go to **Security** and turn on **2-Step Verification**. App passwords require it.
3. Open https://myaccount.google.com/apppasswords, create one (name it "PulmoScan"), and copy the 16 characters.
4. Put them in `MAIL_PASSWORD` **without spaces**.

#### Set up "Sign in with Google" (optional)

1. Go to https://console.cloud.google.com and create a project.
2. **APIs & Services → OAuth consent screen**: choose *External*, then fill in the app name and support email. While the app is in *Testing*, add your own Google account under *Test users*.
3. **APIs & Services → Credentials → Create credentials → OAuth client ID** → *Web application*.
4. Under **Authorized redirect URIs**, add `http://127.0.0.1:5000/google/auth`.
5. Copy the client ID and secret into `.env`.

Open the app at `http://127.0.0.1:5000`, not `http://localhost:5000`. Google only accepts the exact address you registered.

Without these keys, the Google button shows a polite "not configured" message. Everything else works.

### 5. Start the app

```bash
flask --app app run --debug
```

Open **http://127.0.0.1:5000**. Press `Ctrl+C` to stop.

- Use `flask run`, **not** `python app.py`. Only `flask run` reads `.env`. `python app.py` crashes with `KeyError: 'SECRET_KEY'`.
- `--debug` restarts the server whenever you save a file.
- The database and the `instance/uploads/` folder are created automatically on first start. Each computer has its own data, so a fresh clone starts empty.

**Every time after the first setup**, you only need:
```bash
cd Web_Development_Project/projectFinal
source .venv/Scripts/activate      # PowerShell: .venv\Scripts\Activate.ps1
flask --app app run --debug
```

### 6. Run the tests

```bash
python -m unittest test_app
```

All tests should print `OK`. They check that:
- a doctor can't see another doctor's patients,
- only images can be uploaded, and an unreadable image is never labelled "Normal",
- the dashboard shows no made-up accuracy numbers,
- signup codes stay secret and lock after 5 wrong tries,
- the patient form rejects bad input.

---

## The chatbot: Ollama and alternatives

The assistant on the dashboard sends the conversation to an LLM (a large language model) in Ollama's `/api/chat` format. It includes the patient's name, age and latest result. If the LLM can't be reached within 60 seconds, the chat replies *"Assistant interface offline."* The rest of the app works normally.

Pick **one** option.

### Option A: Ollama app on your computer (local development)

1. Install Ollama from https://ollama.com/download and start it.
2. Choose a model:
   - **Cloud model** (runs on Ollama's servers, nothing to download): run `ollama signin` once, then set `OLLAMA_MODEL=gemma4:31b-cloud`. With the local app, cloud models need the **`-cloud`** tag.
   - **Fully offline model** (runs on your computer, needs about 4+ GB RAM): `ollama pull llama3.2`, then set `OLLAMA_MODEL=llama3.2`.
3. Keep `OLLAMA_URL=http://127.0.0.1:11434/api/chat` and leave `OLLAMA_API_KEY` empty.

This does **not** work once deployed, because the server has no Ollama app running.

### Option B: Ollama's hosted API (used for deployment)

No Ollama install is needed. The app calls Ollama's cloud directly.

1. Create an account at https://ollama.com, then create an API key at https://ollama.com/settings/keys.
2. Choose a model from https://ollama.com/api/tags. Use the name **exactly as listed, without `-cloud`**: for example `gemma4:31b`, not `gemma4:31b-cloud`.
3. Set these (in `.env` locally, or in the host's environment settings):
   ```
   OLLAMA_URL=https://ollama.com/api/chat
   OLLAMA_MODEL=gemma4:31b
   OLLAMA_API_KEY=<your key>
   ```

Leaving `-cloud` on the model name is the most common reason the deployed chat shows "offline". Check Ollama's pricing and usage limits for your plan.

### Option C: Your own Ollama server

Run Ollama on a separate server, ideally with a GPU, and point `OLLAMA_URL` at it. Data never leaves your servers, but a GPU server is expensive. Don't expose Ollama's port to the internet without a firewall or authentication in front of it.

### Option D: Another LLM provider

To use a different provider's API (such as Claude or OpenAI), rewrite the `ollama_chat()` function in `app.py` (about 15 lines) to call that provider. Keep its API key in an environment variable, never in code.

### Privacy note

Options A (cloud models), B and D send the patient's name, age and scan result to a third party. For real patient data, check the provider's data-processing terms, or remove the name from `build_system_prompt()` in `app.py`.

---

## Deploy to Render

[Render](https://render.com) is the chosen host. The whole setup is defined in [`projectFinal/render.yaml`](projectFinal/render.yaml), a Render **Blueprint**: one click creates the server, the disk and the settings.

### Why this server size

| Need | Why |
|---|---|
| **2 GB RAM** (plan `1c-2g`) | Measured: about **490 MB** after startup and **505 MB** at peak during a prediction. That is over every 512 MB plan, so those would crash. 2 GB leaves room. |
| **A persistent disk** at `/var/data` | It holds the database and uploaded scans. Without it, **every deploy erases all accounts and scans.** |
| **One worker** (`--workers 1 --threads 4`) | The rate limiter keeps its counts in memory, and SQLite is a single file. Both need one process. The 4 threads serve several users at once. |
| `--timeout 120` | Leaves room for slow LLM replies, which the chat allows up to 60 s. |
| `OMP_NUM_THREADS=1` | Stops PyTorch and scikit-learn from starting a thread per host CPU on a 1-CPU instance. |

**Render's free plan won't work:** it has 512 MB RAM, no disk, and blocks outgoing email (SMTP).

### What `render.yaml` sets up

- A Python web service called `pulmoscan`, plan `1c-2g` (1 CPU, 2 GB RAM), region Singapore.
- `rootDir: projectFinal`, so Render builds inside the app folder.
- Build command `pip install -r requirements.txt`. Start command `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`.
- A 1 GB disk mounted at `/var/data`. Disks can be made larger later, never smaller.
- Health check on `/`. Auto-deploy on every push to the deployed branch.
- These settings, filled in automatically: `PYTHON_VERSION=3.13.16`, `DATA_DIR=/var/data`, `OMP_NUM_THREADS=1`, `OLLAMA_URL=https://ollama.com/api/chat`, and a randomly generated `SECRET_KEY`.
- These you type in yourself (they're secrets, so they're not in the file): `MAIL_USERNAME`, `MAIL_PASSWORD`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `OLLAMA_MODEL`, `OLLAMA_API_KEY`.

### Step 0: Before you start

- [ ] **Revoke the old Gmail app password.** An earlier commit leaked it publicly. Create a new one (see [Get a Gmail app password](#get-a-gmail-app-password)).
- [ ] **Make the GitHub repository private** if its history contains secrets or real patient data (repo owner: *Settings → Danger Zone → Change visibility*).
- [ ] **Merge the deploy-ready code into the branch you'll deploy** (e.g. `main`). Render deploys only what's on that branch.
- [ ] Run `python -m unittest test_app` locally and make sure it prints `OK`.
- [ ] Get an Ollama API key and pick a model name ([Option B](#option-b-ollamas-hosted-api-used-for-deployment)).

### Step 1: Create the Blueprint

1. Sign up at https://render.com with GitHub. A free **Hobby** workspace is enough; you only pay for the server and disk.
2. **New → Blueprint**, then pick this repository and the branch you deploy.
3. **Blueprint file path:** `projectFinal/render.yaml`. The file isn't at the repo root, so you must enter this path.
4. Render shows the service and disk it will create. Fill in the secrets:

| Key | Value |
|---|---|
| `MAIL_USERNAME` | the Gmail address |
| `MAIL_PASSWORD` | the **new** Gmail app password |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | from Google Cloud (see [Set up "Sign in with Google"](#set-up-sign-in-with-google-optional)) |
| `OLLAMA_MODEL` | e.g. `gemma4:31b`, **without** `-cloud` |
| `OLLAMA_API_KEY` | your Ollama API key |

5. Click **Apply**. The first build takes about 2–10 minutes while PyTorch installs. Watch the **Logs** tab. It's done when gunicorn prints `Listening at: http://0.0.0.0:...` and the service shows **Live**.

Your app's address is `https://pulmoscan.onrender.com`, or similar if that name is taken; Render shows it at the top of the service page.

**Keep the generated `SECRET_KEY`.** Changing it logs everyone out.

### Step 2: Allow Google sign-in from the live site

In Google Cloud: **APIs & Services → Credentials →** your OAuth client:
- Add the redirect URI `https://<your-render-url>/google/auth`.
- On the **OAuth consent screen**, click **Publish app**. Until you do, only the test users you listed can sign in.

### Step 3: Check that everything works

- [ ] The home page and About page load.
- [ ] **Sign up** with a real email. The code arrives, and verifying it creates the account.
- [ ] **Log in with Google.**
- [ ] Add a patient and **upload a CT scan**. A result appears.
- [ ] Rename a text file to `test.png` and upload it. You should see *"Could not analyse this image"*, **not** "Normal".
- [ ] Send a chat message. You get a real reply, not "offline".
- [ ] Click **Manual Deploy → Deploy latest commit**, then log in again. Your account and scans are still there. That proves the disk works.

### Step 4: Custom domain (optional)

In the service: **Settings → Custom Domains → Add**, then create the DNS record Render shows you at your domain registrar. Afterwards, add `https://<your-domain>/google/auth` as another redirect URI in Google Cloud.

### Known limits

- **A few seconds of downtime on each deploy.** A service with a disk can't run old and new versions side by side.
- **One instance only.** A disk can't be shared, so the app can't scale out to several servers. One 2 GB instance is plenty for a small clinic or a demo.
- **Not yet verified: does the rate limiter see each visitor's real IP address behind Render's proxy?** (The app trusts one proxy hop: ProxyFix `x_for=1`.) If it doesn't, all visitors share one limit, and a burst of failed logins from anyone could briefly block everyone. To check: temporarily log `request.remote_addr` and `request.headers.get("X-Forwarded-For")` in a route, open the site from two different networks, and compare.

---

## Running costs and day-to-day operations

Prices as of November 2026. Check https://render.com/pricing before you rely on them.

| Item | Cost |
|---|---|
| Web service, plan `1c-2g` (1 CPU, 2 GB RAM) | about **$25 / month** |
| 1 GB persistent disk | about **$0.25 / month** |
| Hobby workspace | $0 (the paid "Pro" workspace is for teams and isn't needed) |
| **Total** | **≈ $25.25 / month** (about $0.035 / hour), billed per second and invoiced monthly |

### Pause, resume, delete

| Action | Where | What happens |
|---|---|---|
| **Suspend** | Service → Settings → Suspend | Server stops and compute billing stops. **Data is kept.** You pay only the disk (~$0.25/month). |
| **Resume** | Service → Resume | Back online in about a minute, at the same URL, with all data. |
| **Delete** | Service → Settings → Delete | **Permanently erases the service and all data on the disk.** Download a backup first. Delete the service, *then* the Blueprint, so a later push doesn't re-create it. |

### Updating the live site

Push to the deployed branch, and Render rebuilds and redeploys automatically. The database and scans on the disk are kept.

### Backups

The whole database is one file, `/var/data/pulmoscan.db`, and scans are in `/var/data/uploads/`. Download copies regularly through the service's **Shell** tab, or check whether your plan includes disk snapshots.

---

## Other hosting options

The app runs on any Linux host that has **≥ 1 GB RAM (2 GB recommended), a persistent disk, and HTTPS**.

| Host | Fit | Notes |
|---|---|---|
| **Render** | ✅ chosen | Simplest. Set up by `render.yaml` above. |
| **DigitalOcean Droplet** (2 GB) | ✅ | About $12/month, but you manage the server. Use a *Droplet*, not *App Platform*, which has no persistent disk. |
| **Oracle Cloud Always Free** (ARM VM, up to 24 GB RAM) | ✅ free | You manage the server. Free capacity is often hard to get. |
| **Railway** | ✅ | Similar to Render. Add a *Volume* and set `DATA_DIR` to its mount path. |
| **Hugging Face Spaces** | ⚠️ needs changes | 16 GB RAM, but no free persistent storage and SMTP is blocked. Would need code changes for storage and email. |
| **Vercel / Netlify** | ❌ | Serverless: 250 MB function limit (PyTorch alone is bigger) and no disk. |

**On your own server (Droplet, Oracle, any VPS):** install Python 3.13, clone the repo, create the venv, and run the same gunicorn command as a `systemd` service. Put **exactly one** HTTPS reverse proxy in front of it (Caddy, or nginx with Let's Encrypt). The app trusts one proxy's `X-Forwarded-*` headers, so don't stack two proxies and don't expose gunicorn directly.

**Email warning:** many hosts block outgoing email (SMTP) on free or trial plans. If verification codes never arrive, upgrade the plan or switch to an email API such as Brevo or Resend.

---

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `KeyError: 'SECRET_KEY'` on start | `.env` is missing or `SECRET_KEY` is empty. Locally, start with `flask --app app run`, not `python app.py`. |
| `Import "flask" could not be resolved` in VS Code | VS Code is using the wrong Python. `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv`. |
| Render build fails: `Could not open requirements file` | `rootDir: projectFinal` is missing from `render.yaml`. |
| *"Failed to dispatch registration email"* | Wrong `MAIL_USERNAME`/`MAIL_PASSWORD`, the app password was revoked, or the host blocks SMTP. |
| Google: `redirect_uri_mismatch` | The URI in Google Cloud must match exactly: `http://127.0.0.1:5000/google/auth` locally, `https://<domain>/google/auth` live. |
| Google: only some people can sign in | The OAuth consent screen is still in *Testing*. Publish it. |
| Chat says *"Assistant interface offline"* | Local: the Ollama app isn't running, or you haven't run `ollama signin`. Deployed: wrong key, or the model name still has `-cloud`. |
| *"Could not analyse this image"* | The file isn't a readable PNG/JPG CT scan. Nothing was saved, by design. |
| *"413 Request Entity Too Large"* | Uploads are limited to 10 MB. |
| *"429 Too Many Requests"* | Rate limit hit (e.g. 10 logins a minute). Wait one minute. |
| All accounts disappear after a deploy | `DATA_DIR` isn't set, or there's no disk. |
| Server restarts during an upload | Out of memory. Use the 2 GB plan and keep `--workers 1`. |
| App won't start: `FileNotFoundError ... .pth/.pkl` | A model file is missing from `projectFinal/`. The app refuses to run without its model, by design. |

---

## Security

**Built in:**
- `SECRET_KEY` has no fallback. The app refuses to start without it, so it can never run with a guessable key.
- Uploaded scans are stored outside `static/` and can't be reached by any URL. Only PNG/JPG up to 10 MB are accepted, and each file gets a random name prefix.
- Signup codes are stored hashed on the server (never in the browser cookie), expire after 10 minutes, and lock after 5 wrong tries.
- Rate limits per IP address: signup 5/min, login 10/min, code check 10/min, code resend 3/min.
- Cookies are `SameSite=Lax`, which blocks other sites from submitting forms as a logged-in doctor. They're `Secure` (HTTPS only) everywhere except local debug mode.
- ProxyFix lets the app build correct `https://` links behind Render's proxy.
- Each doctor can only see and chat about their own patients.

**Your responsibilities:**
- **Never commit** `.env`, `instance/` (database and patient scans), or API keys. `.gitignore` covers these, but check `git status` before every commit.
- If a secret is ever committed, **revoke it immediately**. Deleting it from the code doesn't remove it from git history.
- The model needs three files in `projectFinal/`, and all three must come from the **same** training run: `feature_extractor.pth`, `scaler.pkl`, `svm_model.pkl`. If you retrain, run `AI_End/lung.ipynb` top to bottom, replace all three together, and update `MODEL_TEST_ACCURACY` in `app.py` with the new test score.
