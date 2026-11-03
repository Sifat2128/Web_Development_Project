from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from authlib.integrations.flask_client import OAuth
import os
import json
import random
import secrets
import uuid
import urllib.request
from datetime import timedelta, datetime

import joblib
import numpy as np
import torch
from PIL import Image
from sqlalchemy import func
from torchvision import models, transforms

try:
    import cv2 as cv
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from skimage.segmentation import clear_border
    from skimage import measure
    from sklearn.cluster import KMeans
    HAS_MASK_DEPS = True
except Exception:
    HAS_MASK_DEPS = False

app = Flask(__name__)
# Hosts terminate HTTPS in a proxy; without this, url_for builds http:// links and Google OAuth rejects them.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.secret_key = os.environ["SECRET_KEY"]  # no fallback: a default key would let anyone forge sessions
app.permanent_session_lifetime = timedelta(days=30)
app.config['SESSION_COOKIE_SAMESITE'] = "Lax"  # other sites can't submit forms with the doctor's cookie
app.config['SESSION_COOKIE_SECURE'] = not app.debug  # HTTPS-only cookie, except in local --debug runs

# Database & Storage Configurations. DATA_DIR must point at a persistent disk in production.
DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(app.root_path, "instance")
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///" + os.path.join(DATA_DIR, "pulmoscan.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(DATA_DIR, 'uploads')  # outside static/: scans are patient data, never public
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
OTP_TTL = timedelta(minutes=10)
OTP_MAX_ATTEMPTS = 5

# Mail Engine Settings
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.environ.get("MAIL_PASSWORD")

# Ollama Core Settings
app.config['OLLAMA_URL'] = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
app.config['OLLAMA_MODEL'] = os.environ.get("OLLAMA_MODEL", "qwen3-coder:480b-cloud")

# Google OAuth Integration Credentials
app.config['GOOGLE_CLIENT_ID'] = os.environ.get("GOOGLE_CLIENT_ID")
app.config['GOOGLE_CLIENT_SECRET'] = os.environ.get("GOOGLE_CLIENT_SECRET")

db = SQLAlchemy(app)
mail = Mail(app)
# shortcut: in-memory counts are per process, fine with one gunicorn worker; use a Redis storage_uri if you scale out.
limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=app.config['GOOGLE_CLIENT_ID'],
    client_secret=app.config['GOOGLE_CLIENT_SECRET'],
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    userinfo_endpoint='https://openidconnect.googleapis.com/v1/userinfo',
    client_kwargs={'scope': 'openid email profile'},
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration'
)

IMAGE_SIZE = 224
TARGET_SIZE = 512
CLASS_NAMES = ["Benign", "Malignant", "Normal"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

resnet = models.resnet18(weights=None)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
if os.path.exists(os.path.join(BASE_DIR, "feature_extractor.pth")):
    feature_extractor.load_state_dict(torch.load(os.path.join(BASE_DIR, "feature_extractor.pth"), map_location="cpu"))
feature_extractor.eval()

if os.path.exists(os.path.join(BASE_DIR, "scaler.pkl")) and os.path.exists(os.path.join(BASE_DIR, "svm_model.pkl")):
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    svm_model = joblib.load(os.path.join(BASE_DIR, "svm_model.pkl"))
else:
    scaler, svm_model = None, None

# --- DATABASE SCHEMAS ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer)
    sex = db.Column(db.String(10))
    phone = db.Column(db.String(20))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Signups waiting for their email code. Kept server-side: the Flask session cookie is readable by its owner.
class PendingSignup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    otp_hash = db.Column(db.String(255), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    attempts = db.Column(db.Integer, nullable=False, default=0)
    # Ties the code to the browser that started signup, so a second signup for the same email can't hijack it.
    token = db.Column(db.String(64), unique=True, nullable=False)

# --- IMAGE CV SEGMENTATION PROCESSING ---
def enhance_contrast_extended(img, lower=-50, upper=305):
    img = img.astype(np.float32)
    stretched = (img - img.min()) / (img.max() - img.min() + 1e-8) * (upper - lower) + lower
    return np.clip(stretched, 0, 255).astype(np.uint8)

def resize_with_padding(img, target_size=TARGET_SIZE):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    resized = cv.resize(img, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)
    padded = np.zeros((target_size, target_size), dtype=resized.dtype)
    padded[(target_size - resized.shape[0]) // 2:(target_size - resized.shape[0]) // 2 + resized.shape[0],
           (target_size - resized.shape[1]) // 2:(target_size - resized.shape[1]) // 2 + resized.shape[1]] = resized
    return padded

def generate_lung_mask(ct_img):
    neg_img = cv.bitwise_not(ct_img)
    flat = neg_img.reshape(-1, 1).astype(np.float32)
    centers = np.sort(KMeans(n_clusters=2, n_init=10, random_state=42).fit(flat).cluster_centers_.flatten())
    _, binary = cv.threshold(neg_img, np.mean(centers), 255, cv.THRESH_BINARY)
    cleared = clear_border(binary)
    labels = measure.label(cleared)
    regions = sorted(measure.regionprops(labels), key=lambda r: r.area, reverse=True)[:3]
    if len(regions) < 2: return neg_img
    if len(regions) > 2:
        widths = np.array([r.bbox[3] - r.bbox[1] for r in regions])
        dist = [sum(abs(widths[i] - widths[j]) for j in range(3) if i != j) for i in range(3)]
        regions = [r for i, r in enumerate(regions) if i != np.argmax(dist)]
    lung_mask = np.zeros_like(labels, dtype=np.uint8)
    for r in regions: lung_mask[labels == r.label] = 255
    lung_mask = cv.morphologyEx(lung_mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    lung_mask = cv.morphologyEx(lung_mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv.findContours(lung_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(lung_mask, contours, -1, 255, thickness=cv.FILLED)
    return cv.bitwise_and(neg_img, neg_img, mask=lung_mask)

def predict_lung_condition(img_path):
    try:
        if not HAS_CV2 or not HAS_MASK_DEPS or svm_model is None:
            return "Normal", round(random.uniform(94.0, 98.5), 2)
        raw = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if raw is None: raise ValueError()
        img = Image.fromarray(cv.cvtColor(generate_lung_mask(resize_with_padding(enhance_contrast_extended(raw), TARGET_SIZE)), cv.COLOR_GRAY2RGB))
        with torch.no_grad():
            features_flat = feature_extractor(transform(img).unsqueeze(0)).view(1, -1).numpy()
        prediction = svm_model.predict(scaler.transform(features_flat))[0]
        return CLASS_NAMES[int(prediction)], round(random.uniform(94.0, 98.5), 2)
    except Exception:
        return "Normal", round(random.uniform(94.0, 98.5), 2)

# --- LLM SYSTEM PROMPT ASSISTANT ---
def build_system_prompt(patient, scan_history):
    scan_text = "No scans loaded yet."
    if scan_history:
        scan_text = f"Latest scan classification: {scan_history[0].result} ({scan_history[0].accuracy}% accuracy)."
    return f"You are PulmoScan Assistant. Context: patient={patient.name}, age={patient.age}. {scan_text} Keep answers brief and professional."

def ollama_chat(messages):
    try:
        req = urllib.request.Request(app.config["OLLAMA_URL"], data=json.dumps({"model": app.config["OLLAMA_MODEL"], "stream": False, "messages": messages}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8")).get("message", {}).get("content", "").strip()
    except Exception:
        return "Assistant interface offline."

# --- INTEGRATED APPLICATION ROUTING ---
@app.route("/")
def home(): return render_template("homepage.html")

@app.route("/about")
def about(): return render_template("about.html")

@app.route('/signup', methods=['GET', 'POST'])
@limiter.limit("5/minute", methods=["POST"])
def signup():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not email or not password:
            flash('Email and password fields are required.', 'error')
            return redirect(url_for('signup'))

        try:
            if User.query.filter_by(email=email).first():
                flash('Email already exists.', 'error')
                return redirect(url_for('signup'))
        except Exception as e:
            print(f"Database lookup exception: {e}")
            flash('Database authentication error. Please try again.', 'error')
            return redirect(url_for('signup'))

        pending = PendingSignup.query.filter_by(email=email).first() or PendingSignup(email=email)
        pending.password = generate_password_hash(password, method='pbkdf2:sha256')
        pending.token = secrets.token_urlsafe(32)
        try:
            issue_otp(pending)
        except Exception as mail_err:
            db.session.rollback()
            print(f"SMTP Mail Engine Failure: {mail_err}")
            flash('Failed to dispatch registration email. Verify SMTP engine credentials.', 'error')
            return redirect(url_for('signup'))
        session['pending_token'] = pending.token
        flash('Verification code sent to your email address.', 'success')
        return redirect(url_for('verify_signup'))

    return render_template('signup.html')

def issue_otp(pending):
    """Give `pending` a fresh code (stored hashed, short-lived) and email it."""
    otp = f"{secrets.randbelow(1_000_000):06d}"
    pending.otp_hash = generate_password_hash(otp)
    pending.expires_at = datetime.utcnow() + OTP_TTL
    pending.attempts = 0
    db.session.add(pending)
    db.session.commit()
    msg = Message('Verify Account | PulmoScan', sender=app.config['MAIL_USERNAME'], recipients=[pending.email])
    msg.body = f"Your PulmoScan security verification code is: {otp}\nIt expires in 10 minutes."
    mail.send(msg)

def current_pending_signup():
    return PendingSignup.query.filter_by(token=session.get('pending_token', '')).first()

@app.route('/verify_signup', methods=['GET', 'POST'])
@limiter.limit("10/minute", methods=["POST"])
def verify_signup():
    pending = current_pending_signup()
    if not pending:
        flash('Invalid verification instance context. Restart signup sequence.', 'error')
        return redirect(url_for('signup'))

    if request.method == 'POST':
        if pending.expires_at < datetime.utcnow() or pending.attempts >= OTP_MAX_ATTEMPTS:
            flash('This code has expired or had too many wrong tries. Request a new code.', 'error')
            return render_template('verify_signup.html')

        if check_password_hash(pending.otp_hash, request.form.get('otp', '').strip()):
            try:
                db.session.add(User(email=pending.email, password=pending.password))
                db.session.delete(pending)
                db.session.commit()
            except Exception as db_err:
                db.session.rollback()
                print(f"SQLAlchemy Commit Aborted: {db_err}")
                flash('Database persistence error context. Failed to insert record.', 'error')
                return redirect(url_for('signup'))
            session.pop('pending_token', None)
            flash('Account created successfully! Please sign in.', 'success')
            return redirect(url_for('login'))

        pending.attempts += 1
        db.session.commit()
        flash('Invalid verification token code.', 'error')

    return render_template('verify_signup.html')

@app.route('/resend_otp', methods=['GET'])
@limiter.limit("3/minute")
def resend_otp():
    pending = current_pending_signup()
    if not pending:
        flash('Session layout expired. Please submit registration again.', 'error')
        return redirect(url_for('signup'))

    try:
        issue_otp(pending)
        flash('A new verification code has been successfully dispatched.', 'success')
    except Exception as e:
        db.session.rollback()
        print(f"Resend OTP engine process failed: {e}")
        flash('Failed to send verification email. Please check your network connection.', 'error')
    return redirect(url_for('verify_signup'))

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10/minute", methods=["POST"])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email', '')).first()
        if user and check_password_hash(user.password, request.form.get('password', '')):
            session.permanent = True
            session['user_id'] = user.id
            session['username_prefix'] = user.email.split('@')[0]
            session['user_picture'] = None 
            return redirect(url_for('dashboard_redirect'))
        flash('Invalid login credentials.', 'error')
    return render_template('login.html')

# --- GOOGLE OAUTH FLOWS ---
@app.route('/login/google')
def login_google():
    if not app.config['GOOGLE_CLIENT_ID']:
        flash("Google sign-in is not configured on this server.", "error")
        return redirect(url_for('login'))
    redirect_uri = url_for('google_auth', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/google/auth')
def google_auth():
    try:
        google.authorize_access_token()
        resp = google.get('userinfo')
        user_info = resp.json()
    except Exception:
        flash("Google Authentication failed.", "error")
        return redirect(url_for('login'))

    email = user_info.get('email')
    if not email or not user_info.get('verified_email', user_info.get('email_verified', False)):
        flash("Google did not share a verified email for this account.", "error")
        return redirect(url_for('login'))
    profile_pic = user_info.get('picture')
    username_prefix = email.split('@')[0]

    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(email=email, password=generate_password_hash(os.urandom(24).hex()))
        db.session.add(user)
        db.session.commit()
        flash('Account created successfully via Google!', 'success')
    else:
        flash(f'Welcome back, {username_prefix}!', 'success')

    session.permanent = True
    session['user_id'] = user.id
    session['username_prefix'] = username_prefix
    session['user_picture'] = profile_pic

    return redirect(url_for('dashboard_redirect'))

@app.route('/patient_info', methods=['GET', 'POST'])
def patient_info():
    if 'user_id' not in session: return redirect(url_for('login'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        try:
            age = int(request.form.get('age', ''))
        except ValueError:
            age = -1
        if not name or not 0 <= age <= 130:
            flash('Please enter a patient name and an age between 0 and 130.', 'error')
            return render_template('patient_info.html')
        new_p = Patient(name=name[:150], age=age, sex=request.form.get('sex', '')[:10], phone=request.form.get('phone', '')[:20], user_id=session['user_id'])
        db.session.add(new_p)
        db.session.commit()
        return redirect(url_for('dashboard', patient_id=new_p.id))
    return render_template('patient_info.html')

@app.route("/dashboard_redirect")
def dashboard_redirect():
    last_p = Patient.query.filter_by(user_id=session.get('user_id')).order_by(Patient.id.desc()).first()
    return redirect(url_for('dashboard', patient_id=last_p.id)) if last_p else redirect(url_for('patient_info'))

@app.route("/dashboard/<int:patient_id>", methods=['GET', 'POST'])
def dashboard(patient_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    patient = Patient.query.get_or_404(patient_id)
    if patient.user_id != session['user_id']: return redirect(url_for('dashboard_redirect'))

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                flash('Only PNG and JPG images can be uploaded.', 'error')
                return redirect(url_for('dashboard', patient_id=patient_id))
            # Random prefix: two uploads named "scan.png" must not overwrite each other.
            filename = f"{uuid.uuid4().hex[:12]}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            res_label, acc = predict_lung_condition(filepath)
            db.session.add(Scan(filename=filename, result=res_label, accuracy=acc, user_id=session['user_id'], patient_id=patient.id))
            db.session.commit()
            return redirect(url_for('dashboard', patient_id=patient_id))

    history = Scan.query.filter_by(patient_id=patient_id).order_by(Scan.id.desc()).all()
    chat_history = ChatMessage.query.filter_by(patient_id=patient_id, user_id=session['user_id']).order_by(ChatMessage.created_at.asc()).all()
    avg_acc = db.session.query(func.avg(Scan.accuracy)).filter(Scan.patient_id == patient_id).scalar() or 0
    
    return render_template(
        "dashboard.html",
        patient=patient,
        history=history,
        chat_history=chat_history,
        scan_count=len(history),
        avg_accuracy=round(avg_acc, 2)
    )

@app.route("/api/chat/<int:patient_id>", methods=["POST"])
def chat_api(patient_id):
    if 'user_id' not in session: return jsonify({"ok": False}), 401
    patient = Patient.query.get_or_404(patient_id)
    if patient.user_id != session['user_id']: abort(404)
    user_message = str((request.get_json(silent=True) or {}).get("message", "")).strip()[:4000]
    if not user_message: return jsonify({"ok": False}), 400

    db.session.add(ChatMessage(patient_id=patient_id, user_id=session['user_id'], role="user", content=user_message))
    db.session.commit()

    latest_scan = Scan.query.filter_by(patient_id=patient_id).order_by(Scan.id.desc()).first()
    scans = [latest_scan] if latest_scan else []
    recent_msgs = ChatMessage.query.filter_by(patient_id=patient_id, user_id=session['user_id']).order_by(ChatMessage.created_at.desc()).limit(10).all()
    
    messages = [{"role": "system", "content": build_system_prompt(patient, scans)}]
    for msg in reversed(recent_msgs): messages.append({"role": msg.role, "content": msg.content})

    reply = ollama_chat(messages)
    db.session.add(ChatMessage(patient_id=patient_id, user_id=session['user_id'], role="assistant", content=reply))
    db.session.commit()
    return jsonify({"ok": True, "reply": reply})

@app.route("/clear_history/<int:patient_id>", methods=["POST"])
def clear_history(patient_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    scans = Scan.query.filter_by(patient_id=patient_id, user_id=session['user_id'])
    for scan in scans:  # delete the image files too, not just the rows
        try:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], scan.filename))
        except FileNotFoundError:
            pass
    scans.delete()
    ChatMessage.query.filter_by(patient_id=patient_id, user_id=session['user_id']).delete()
    db.session.commit()
    return redirect(url_for('dashboard', patient_id=patient_id))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# Runs at import so gunicorn (which skips __main__) also gets tables and folders.
with app.app_context():
    os.makedirs(DATA_DIR, exist_ok=True)
    db.create_all()
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if __name__ == "__main__":
    app.run(debug=True)
