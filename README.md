# PulmoScan

A web app that helps doctors screen lung CT scans. A doctor signs up, adds patients, and uploads scans. A ResNet18 + SVM model labels each scan **Benign**, **Malignant** or **Normal**. An AI assistant answers questions about the patient's results.

> **Medical disclaimer:** The model is 93.6% accurate on its test set (1,478 images). It is a research tool, not a diagnosis. A clinician must confirm every result.

---

## Contents

1. [Repository layout](#repository-layout)
2. [Run locally](#run-locally)
3. [The chatbot: Ollama and alternatives](#the-chatbot-ollama-and-alternatives)
4. [Deploy to production (Render)](#deploy-to-production-render)
5. [Other hosts](#other-hosts)
6. [Troubleshooting](#troubleshooting)
7. [Security rules](#security-rules)

---

## Repository layout

```
Web_Development_Project/
├── projectFinal/            ← the web app (this is what you run and deploy)
│   ├── app.py               ← Flask app: routes, database, model inference, chatbot
│   ├── test_app.py          ← safety tests (python -m unittest test_app)
│   ├── requirements.txt     ← Python packages
│   ├── .env.example         ← template for your settings (copy to .env)
│   ├── feature_extractor.pth, scaler.pkl, svm_model.pkl   ← trained model files
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
| Python | 3.13 | https://www.python.org/downloads/ (tick **"Add Python to PATH"** on Windows) |
| Git | any | https://git-scm.com/downloads |
| Ollama | optional, only for the chatbot | https://ollama.com/download |

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

Your prompt now starts with `(.venv)`. The install takes a few minutes because PyTorch is large. It installs the CPU-only build, which is all the app needs.

### 4. Create your settings file

```bash
cp .env.example .env               # PowerShell: Copy-Item .env.example .env
```

Open `.env` and fill it in. **Never commit `.env`.** It is already in `.gitignore`.

| Setting | Required? | What to put |
|---|---|---|
| `SECRET_KEY` | **Yes.** The app won't start without it. | A long random string. Generate one: `python -c "import secrets; print(secrets.token_hex(32))"` |
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

Open **http://127.0.0.1:5000**. `flask run` reads `.env` automatically, and `--debug` restarts the server whenever you save a file. Press `Ctrl+C` to stop.

**Every time after the first setup**, you only need:
```bash
cd Web_Development_Project/projectFinal
source .venv/Scripts/activate      # PowerShell: .venv\Scripts\Activate.ps1
flask --app app run --debug
```

The database and the `instance/uploads/` folder are created automatically on first start. Each computer has its own data, so a fresh clone starts empty.

### 6. Run the tests

```bash
python -m unittest test_app
```

All tests should print `OK`. They check that:
- a doctor can't see another doctor's patients,
- only images can be uploaded,
- an unreadable image is never labelled "Normal",
- signup codes stay secret and lock after 5 wrong tries,
- the patient form rejects bad input.

---

## The chatbot: Ollama and alternatives

The assistant on the dashboard sends the conversation to an LLM (a large language model) through Ollama's `/api/chat` format. It includes the patient's name, age and latest result. If the LLM can't be reached, the chat replies *"Assistant interface offline."* The rest of the app works normally.

Pick **one** option.

### Option A: Ollama app on your computer (local development)

1. Install Ollama from https://ollama.com/download and start it.
2. Choose a model:
   - **Cloud model** (runs on Ollama's servers, nothing to download): run `ollama signin` once. The default `OLLAMA_MODEL=qwen3-coder:480b-cloud` then works.
   - **Fully offline model** (runs on your computer, needs about 4+ GB RAM): `ollama pull llama3.2`, then set `OLLAMA_MODEL=llama3.2`.
3. Keep `OLLAMA_URL=http://127.0.0.1:11434/api/chat` and leave `OLLAMA_API_KEY` empty.

This does **not** work once deployed, because the server has no Ollama app running.

### Option B: Ollama's hosted API (recommended for deployment)

No Ollama install is needed. The app calls Ollama's cloud directly.

1. Create an account at https://ollama.com, then create an API key at https://ollama.com/settings/keys.
2. Choose a model from the list at https://ollama.com/api/tags. Use the name **exactly as listed, without** `-cloud`.
3. Set these (in `.env` locally, or in the host's environment settings):
   ```
   OLLAMA_URL=https://ollama.com/api/chat
   OLLAMA_MODEL=<model name from the list>
   OLLAMA_API_KEY=<your key>
   ```

`qwen3-coder` is built for programming. For a patient assistant, a general chat model from the list fits better. Check Ollama's pricing and usage limits for your plan.

### Option C: Your own Ollama server

Run Ollama on a separate server, ideally with a GPU, and point `OLLAMA_URL` at it. You get full control and data never leaves your servers, but a GPU server is expensive. Don't expose Ollama's port to the internet without a firewall or authentication in front of it.

### Option D: Another LLM provider

To use a different provider's API (such as Claude or OpenAI), rewrite the `ollama_chat()` function in `app.py` (about 15 lines) to call that provider's API. Store the provider's API key in an environment variable, never in code.

### Privacy note

Options A (cloud model), B and D send the patient's name, age and scan result to a third party. For real patient data, check the provider's data-processing terms, or remove the name from `build_system_prompt()` in `app.py`.

---

## Deploy to production (Render)

[Render](https://render.com) is the recommended host. It deploys straight from GitHub, provides HTTPS automatically, and offers a persistent disk. Plans and prices change, so check Render's pricing page.

### What the server needs

| Need | Why |
|---|---|
| **At least 1 GB RAM** | PyTorch and the model use several hundred MB. 512 MB plans will likely crash. |
| **A persistent disk** | It holds the database and uploaded scans. Without one, **every deploy erases all accounts and scans.** Free instances have no disk. |
| **HTTPS** | Login cookies are sent over HTTPS only in production. Render provides HTTPS automatically. |

### Step 0: Before you start

- [ ] **Revoke the old Gmail app password.** An earlier commit leaked it publicly. Create a new one (see above).
- [ ] **Make the GitHub repository private** if it contains real patient data or secrets in its history (repo owner: *Settings → Danger Zone → Change visibility*).
- [ ] Run the tests locally and make sure they pass.

### Step 1: Create the web service

1. Sign up at https://render.com with GitHub.
2. **New → Web Service**, then pick this repository.
3. Fill in:

| Field | Value |
|---|---|
| **Branch** | the branch you deploy (e.g. `main`) |
| **Root Directory** | `projectFinal` |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --workers 1 --threads 4 --timeout 120` |
| **Instance Type** | one with **at least 1 GB RAM** |

Why `--workers 1`: each worker loads its own copy of PyTorch and the model, and the login rate limiter only keeps correct counts within one worker. The 4 threads handle several users at once.

### Step 2: Add a persistent disk

In the service: **Disks → Add Disk**.
- **Mount path:** `/var/data`
- **Size:** 1 GB is plenty to start (scans are about 100–600 KB each).

### Step 3: Set the environment variables

In the service: **Environment → Add Environment Variable**.

| Key | Value |
|---|---|
| `PYTHON_VERSION` | `3.13.1` |
| `SECRET_KEY` | a **new** random string (`python -c "import secrets; print(secrets.token_hex(32))"`) |
| `DATA_DIR` | `/var/data` |
| `MAIL_USERNAME` | the Gmail address |
| `MAIL_PASSWORD` | the **new** Gmail app password |
| `GOOGLE_CLIENT_ID` | from Google Cloud (Step 5) |
| `GOOGLE_CLIENT_SECRET` | from Google Cloud (Step 5) |
| `OLLAMA_URL` | `https://ollama.com/api/chat` ([Option B](#option-b-ollamas-hosted-api-recommended-for-deployment)) |
| `OLLAMA_MODEL` | a model name from https://ollama.com/api/tags |
| `OLLAMA_API_KEY` | your Ollama API key |

**Keep `SECRET_KEY` the same across deploys.** Changing it logs everyone out.

### Step 4: Deploy

Click **Create Web Service** (or **Manual Deploy**). Watch the **Logs** tab. The first build takes several minutes while PyTorch installs. Deployment is done when you see gunicorn's `Listening at: http://0.0.0.0:...` and the service shows **Live**.

Your app's address is `https://<service-name>.onrender.com`.

### Step 5: Allow Google sign-in from the live site

In Google Cloud: **APIs & Services → Credentials →** your OAuth client:
- Add the redirect URI `https://<service-name>.onrender.com/google/auth`.
- On the **OAuth consent screen**, click **Publish app** so anyone can sign in, not just test users.

### Step 6: Check that everything works

- [ ] The home page and About page load.
- [ ] **Sign up** with a real email. The code arrives, and verifying it creates the account.
- [ ] **Log in with Google.**
- [ ] Add a patient and **upload a CT scan**. A result appears.
- [ ] Rename a text file to `test.png` and upload it. You should see *"Could not analyse this image"*, **not** "Normal".
- [ ] Send a chat message. You get a real reply, not "offline".
- [ ] **Redeploy** (Manual Deploy), then log in again. Your account and scans are still there. That proves the disk works.

### Step 7: Custom domain (optional)

In Render: **Settings → Custom Domains → Add**, then create the DNS record Render shows you at your domain registrar. Afterwards, add `https://<your-domain>/google/auth` as another redirect URI in Google Cloud.

### Updating the live site

Push to the deployed branch, and Render rebuilds and redeploys automatically. The database and scans on the disk are kept.

### Backups

The whole database is one file: `/var/data/pulmoscan.db`. Download a copy regularly from Render's **Shell** tab, or check whether your Render plan includes disk snapshots.

---

## Other hosts

The app runs on any Linux host that meets the same three needs (≥ 1 GB RAM, a persistent disk, HTTPS):

- **Railway:** similar to Render. Add a *Volume*, set `DATA_DIR` to its mount path, and use the same start command and variables.
- **Your own VPS** (e.g. Ubuntu): install Python 3.13, clone the repo, create the venv, and run the same gunicorn command as a `systemd` service. Put **one** reverse proxy with HTTPS in front of it (Caddy or nginx + Let's Encrypt). The app trusts exactly one proxy's `X-Forwarded-*` headers, so don't stack two proxies, and don't expose gunicorn directly to the internet.

**Not suitable:** serverless platforms such as Vercel and Netlify (no persistent disk, model too large) and very small free tiers (not enough RAM).

**Email warning:** some hosts block outgoing email (SMTP) on free or trial plans. If verification codes never arrive, upgrade the plan or switch to an email API such as Brevo or Resend.

---

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `KeyError: 'SECRET_KEY'` on start | `.env` is missing, or `SECRET_KEY` is empty. Locally, start with `flask --app app run`, not `python app.py` (only `flask run` reads `.env`). |
| `Import "flask" could not be resolved` in VS Code | VS Code is using the wrong Python. `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv`. |
| *"Failed to dispatch registration email"* | Wrong `MAIL_USERNAME`/`MAIL_PASSWORD`, the app password was revoked, or the host blocks SMTP. |
| Google: `redirect_uri_mismatch` | The URI in Google Cloud must match exactly: `http://127.0.0.1:5000/google/auth` locally, `https://<domain>/google/auth` live. |
| Chat says *"Assistant interface offline"* | The Ollama app isn't running (Option A), or the URL, model name or key is wrong (Option B). Model names for the hosted API come from https://ollama.com/api/tags, with no `-cloud`. |
| *"Could not analyse this image"* | The file isn't a readable PNG/JPG CT scan. Nothing was saved, by design. |
| *"413 Request Entity Too Large"* | Uploads are limited to 10 MB. |
| *"429 Too Many Requests"* on login | Rate limit (10 tries a minute). Wait one minute. |
| All accounts disappear after a deploy | `DATA_DIR` isn't set, or there's no persistent disk. See Steps 2–3. |
| Server crashes or restarts during upload | Out of memory. Use an instance with ≥ 1 GB RAM and keep `--workers 1`. |
| App won't start: `FileNotFoundError ... .pth/.pkl` | A model file is missing from `projectFinal/`. The app refuses to run without its model, by design. |

---

## Security rules

- **Never commit** `.env`, `instance/` (database and patient scans), or API keys. `.gitignore` covers these. Check `git status` before every commit anyway.
- If a secret is ever committed, **revoke it immediately**. Deleting it from the code does not remove it from git history.
- Patient scans are stored outside `static/` and are never publicly reachable by URL. Keep it that way.
- The model needs these three files in `projectFinal/`, and they must come from the **same** training run: `feature_extractor.pth`, `scaler.pkl`, `svm_model.pkl`. If you retrain, run `AI_End/lung.ipynb` top to bottom and replace all three together, then update `MODEL_TEST_ACCURACY` in `app.py` with the new test score.
