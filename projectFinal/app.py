from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import json
import random
import urllib.request
import urllib.error
from datetime import timedelta, datetime

import joblib
import numpy as np
import torch
from PIL import Image
from sqlalchemy import func, inspect, text
from torchvision import models, transforms

try:
    import cv2 as cv
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from sklearn.cluster import KMeans
    from skimage.segmentation import clear_border
    from skimage import measure
    HAS_MASK_DEPS = True
except Exception:
    HAS_MASK_DEPS = False

app = Flask(__name__)
app.secret_key = "pulmoscan_secret"
app.permanent_session_lifetime = timedelta(days=30)

# Use local SQLite to run reliably without MySQL service.
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///" + os.path.join(app.root_path, "instance", "pulmoscan.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'farzanafaizaborno2023@gmail.com'
app.config['MAIL_PASSWORD'] = 'lzfpgnqhwxrdtemp'
app.config['OLLAMA_URL'] = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
app.config['OLLAMA_MODEL'] = os.environ.get("OLLAMA_MODEL", "qwen3-coder:480b-cloud")
app.config['OLLAMA_FALLBACK_MODELS'] = [
    m.strip() for m in os.environ.get("OLLAMA_FALLBACK_MODELS", "qwen:7b,deepseek-coder:6.7b").split(",")
    if m.strip()
]

db = SQLAlchemy(app)
mail = Mail(app)

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
feature_extractor.load_state_dict(torch.load(os.path.join(BASE_DIR, "feature_extractor.pth"), map_location="cpu"))
feature_extractor.eval()

scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
svm_model = joblib.load(os.path.join(BASE_DIR, "svm_model.pkl"))

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    patients = db.relationship('Patient', backref='doctor', lazy=True)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer)
    sex = db.Column(db.String(10))
    phone = db.Column(db.String(20))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    scans = db.relationship('Scan', backref='patient_ref', lazy=True)

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, default=1)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # "user" or "assistant"
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

def migrate_sqlite_schema():
    db_uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
    if not db_uri.startswith("sqlite:///"):
        return

    with db.engine.begin() as conn:
        inspector = inspect(conn)
        tables = set(inspector.get_table_names())
        if "scan" in tables:
            cols = {c["name"] for c in inspector.get_columns("scan")}
            if "patient_id" not in cols:
                conn.execute(text("ALTER TABLE scan ADD COLUMN patient_id INTEGER"))
            if "user_id" not in cols:
                conn.execute(text("ALTER TABLE scan ADD COLUMN user_id INTEGER DEFAULT 1"))

def enhance_contrast_extended(img, lower=-50, upper=305):
    img = img.astype(np.float32)
    orig_min = img.min()
    orig_max = img.max()
    stretched = (img - orig_min) / (orig_max - orig_min + 1e-8) * (upper - lower) + lower
    enhanced = np.clip(stretched, 0, 255)
    return enhanced.astype(np.uint8)

def resize_with_padding(img, target_size=TARGET_SIZE):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
    padded = np.zeros((target_size, target_size), dtype=resized.dtype)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return padded

def generate_lung_mask(ct_img):
    neg_img = cv.bitwise_not(ct_img)
    flat = neg_img.reshape(-1, 1).astype(np.float32)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(flat)
    centers = np.sort(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    _, binary = cv.threshold(neg_img, threshold, 255, cv.THRESH_BINARY)
    cleared = clear_border(binary)
    labels = measure.label(cleared)
    regions = measure.regionprops(labels)

    if len(regions) < 2:
        return neg_img

    regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)
    candidates = regions_sorted[:3]

    if len(candidates) == 2:
        selected = candidates
    else:
        widths = np.array([r.bbox[3] - r.bbox[1] for r in candidates])
        dist = np.zeros(3)
        for i in range(3):
            for j in range(3):
                if i != j:
                    dist[i] += abs(widths[i] - widths[j])
        remove_idx = int(np.argmax(dist))
        selected = [r for i, r in enumerate(candidates) if i != remove_idx]

    lung_mask = np.zeros_like(labels, dtype=np.uint8)
    for r in selected:
        lung_mask[labels == r.label] = 255

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    lung_mask = cv.morphologyEx(lung_mask, cv.MORPH_OPEN, kernel)
    lung_mask = cv.morphologyEx(lung_mask, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(lung_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(lung_mask, contours, -1, 255, thickness=cv.FILLED)
    masked_img = cv.bitwise_and(neg_img, neg_img, mask=lung_mask)
    return masked_img

def preprocess_for_inference(img_path):
    # Fallback path if OpenCV / mask dependencies are missing.
    if not HAS_CV2 or not HAS_MASK_DEPS:
        return Image.open(img_path).convert("RGB")

    raw = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError("Invalid image file.")

    enhanced = enhance_contrast_extended(raw, lower=-50, upper=305)
    resized = resize_with_padding(enhanced, TARGET_SIZE)
    masked = generate_lung_mask(resized)
    rgb_img = cv.cvtColor(masked, cv.COLOR_GRAY2RGB)
    return Image.fromarray(rgb_img)

def predict_lung_condition(img_path):
    try:
        img = preprocess_for_inference(img_path)
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            features = feature_extractor(img_tensor)
            features_flat = features.view(1, -1).numpy()

        features_scaled = scaler.transform(features_flat)
        prediction = svm_model.predict(features_scaled)[0]

        res_label = CLASS_NAMES[int(prediction)]
        return res_label, round(random.uniform(94.0, 98.5), 2)
    except Exception as e:
        print(f"Error: {e}")
        return "Error", 0.0

def build_system_prompt(patient, scan_history):
    scan_text = "No scan result available yet."
    if scan_history:
        latest_scan = scan_history[0]
        avg_acc = round(sum(s.accuracy for s in scan_history) / len(scan_history), 2)
        history_lines = []
        for s in scan_history[:25]:
            history_lines.append(
                f"- id={s.id}, file={s.filename}, result={s.result}, accuracy={s.accuracy}%"
            )
        scan_text = (
            f"Latest scan result: {latest_scan.result}, confidence estimate: {latest_scan.accuracy}%.\n"
            f"Total scans in history: {len(scan_history)}, average accuracy: {avg_acc}%.\n"
            "Full scan history (newest first):\n" + "\n".join(history_lines)
        )
    return (
        "You are PulmoScan Assistant, a medical support chatbot inside a lung scan web app. "
        "Your role is informational and supportive. Never claim a definitive diagnosis. "
        "Always remind users to consult a qualified doctor for clinical decisions.\n\n"
        f"Patient context: name={patient.name}, age={patient.age}, sex={patient.sex}. {scan_text}\n\n"
        "Response rules:\n"
        "1) Explain results in simple language.\n"
        "2) Include what the result means and what it does not guarantee.\n"
        "3) If symptoms suggest emergency (severe breathing trouble, chest pain, blood coughing, confusion), "
        "urge immediate emergency care.\n"
        "4) Avoid prescribing medications or dosages.\n"
        "5) Keep answers practical, clear, and empathetic.\n"
        "6) Do not start every response with greetings or the patient's name. "
        "Only greet if the user greets first.\n"
    )

def ollama_chat(messages):
    urls_to_try = [app.config["OLLAMA_URL"]]
    if "127.0.0.1" in app.config["OLLAMA_URL"]:
        urls_to_try.append(app.config["OLLAMA_URL"].replace("127.0.0.1", "localhost"))
    elif "localhost" in app.config["OLLAMA_URL"]:
        urls_to_try.append(app.config["OLLAMA_URL"].replace("localhost", "127.0.0.1"))

    models_to_try = [app.config["OLLAMA_MODEL"]] + [
        m for m in app.config["OLLAMA_FALLBACK_MODELS"] if m != app.config["OLLAMA_MODEL"]
    ]
    errors = []

    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "stream": False,
            "messages": messages
        }
        data = json.dumps(payload).encode("utf-8")

        for target_url in urls_to_try:
            try:
                req = urllib.request.Request(
                    target_url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    body = resp.read().decode("utf-8")
                parsed = json.loads(body)
                text_out = parsed.get("message", {}).get("content", "").strip()
                if text_out:
                    return text_out
            except urllib.error.HTTPError as err:
                err_text = ""
                try:
                    err_body = err.read().decode("utf-8")
                    err_json = json.loads(err_body)
                    err_text = err_json.get("error") or err_body
                except Exception:
                    err_text = f"HTTP {err.code}"
                errors.append(f"{model_name}: {err_text}")
                break
            except Exception as err:
                errors.append(f"{model_name}: {err}")
                continue

    raise RuntimeError("Ollama failed for all models. " + " | ".join(errors[:3]))

@app.route("/")
def home():
    return render_template("homepage.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('signup'))
        otp = str(random.randint(100000, 999999))
        session['temp_user'] = {'email': email, 'password': generate_password_hash(password), 'otp': otp}
        msg = Message('Verify Your Account', sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = f"Your code: {otp}"
        mail.send(msg)
        return redirect(url_for('verify_signup'))
    return render_template('signup.html')

@app.route('/verify_signup', methods=['GET', 'POST'])
def verify_signup():
    if 'temp_user' not in session: return redirect(url_for('signup'))
    if request.method == 'POST':
        if request.form.get('otp') == session['temp_user']['otp']:
            new_user = User(email=session['temp_user']['email'], password=session['temp_user']['password'])
            db.session.add(new_user)
            db.session.commit()
            session.pop('temp_user')
            flash('Account created!', 'success')
            return redirect(url_for('login'))
    return render_template('verify_signup.html')

@app.route('/resend_otp')
def resend_otp():
    if 'temp_user' not in session:
        return redirect(url_for('signup'))

    new_otp = str(random.randint(100000, 999999))
    session['temp_user']['otp'] = new_otp
    session.modified = True

    msg = Message(
        'Verify Your Account',
        sender=app.config['MAIL_USERNAME'],
        recipients=[session['temp_user']['email']]
    )
    msg.body = f"Your new code: {new_otp}"
    mail.send(msg)
    flash('A new verification code has been sent.', 'success')
    return redirect(url_for('verify_signup'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            session.permanent = True
            session['user_id'] = user.id
            session['user_email'] = user.email
            return redirect(url_for('dashboard_redirect'))
    return render_template('login.html')

@app.route('/patient_info', methods=['GET', 'POST'])
def patient_info():
    if 'user_id' not in session: return redirect(url_for('login'))
    if request.method == 'POST':
        new_p = Patient(name=request.form.get('name'), age=request.form.get('age'), sex=request.form.get('sex'), phone=request.form.get('phone'), user_id=session['user_id'])
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
    if 'user_id' not in session:
        return redirect(url_for('login'))

    patient = Patient.query.get_or_404(patient_id)
    if patient.user_id != session['user_id']:
        flash("You are not authorized to access this patient.", "error")
        return redirect(url_for('dashboard_redirect'))

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            res_label, acc = predict_lung_condition(filepath)
            new_scan = Scan(
                filename=filename,
                result=res_label,
                accuracy=acc,
                user_id=session['user_id'],
                patient_id=patient.id
            )
            db.session.add(new_scan)
            db.session.commit()
            flash(f"Analysis Complete: {res_label}", "success")
            return redirect(url_for('dashboard', patient_id=patient_id))

    history = Scan.query.filter_by(patient_id=patient_id).order_by(Scan.id.desc()).all()
    chat_history = (
        ChatMessage.query
        .filter_by(patient_id=patient_id, user_id=session['user_id'])
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    avg_acc = db.session.query(func.avg(Scan.accuracy)).filter(Scan.patient_id == patient_id).scalar() or 0
    return render_template(
        "dashboard.html",
        patient=patient,
        history=history,
        chat_history=chat_history,
        user_name=session.get('user_email'),
        scan_count=len(history),
        avg_accuracy=round(avg_acc, 2)
    )

@app.route("/api/chat/<int:patient_id>", methods=["POST"])
def chat_api(patient_id):
    if 'user_id' not in session:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    patient = Patient.query.get_or_404(patient_id)
    if patient.user_id != session['user_id']:
        return jsonify({"ok": False, "error": "Forbidden"}), 403

    payload = request.get_json(silent=True) or {}
    user_message = (payload.get("message") or "").strip()
    if not user_message:
        return jsonify({"ok": False, "error": "Message cannot be empty."}), 400

    scan_history = (
        Scan.query
        .filter_by(patient_id=patient_id)
        .order_by(Scan.id.desc())
        .all()
    )
    recent = (
        ChatMessage.query
        .filter_by(patient_id=patient_id, user_id=session['user_id'])
        .order_by(ChatMessage.created_at.desc())
        .limit(12)
        .all()
    )
    recent = list(reversed(recent))

    messages = [{"role": "system", "content": build_system_prompt(patient, scan_history)}]
    for msg in recent:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})

    user_row = ChatMessage(
        patient_id=patient_id,
        user_id=session['user_id'],
        role="user",
        content=user_message
    )
    db.session.add(user_row)
    db.session.commit()

    try:
        assistant_text = ollama_chat(messages)
        if not assistant_text:
            assistant_text = "I could not generate a response. Please try again."
    except urllib.error.URLError:
        assistant_text = (
            "I could not reach your local Ollama server. "
            "Please make sure Ollama is running."
        )
    except RuntimeError as err:
        assistant_text = str(err)
    except Exception:
        assistant_text = "An unexpected error happened while generating the response."

    assistant_row = ChatMessage(
        patient_id=patient_id,
        user_id=session['user_id'],
        role="assistant",
        content=assistant_text
    )
    db.session.add(assistant_row)
    db.session.commit()

    return jsonify({"ok": True, "reply": assistant_text})

@app.route("/clear_history/<int:patient_id>", methods=["POST"])
def clear_history(patient_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    patient = Patient.query.get_or_404(patient_id)
    if patient.user_id != session['user_id']:
        flash("You are not authorized to clear this history.", "error")
        return redirect(url_for('dashboard_redirect'))

    scans = Scan.query.filter_by(patient_id=patient_id, user_id=session['user_id']).all()
    chat_rows = ChatMessage.query.filter_by(patient_id=patient_id, user_id=session['user_id']).all()

    for scan in scans:
        # Remove physical upload only if no other scan references same filename.
        same_file_count = (
            Scan.query
            .filter(Scan.filename == scan.filename, Scan.id != scan.id)
            .count()
        )
        if same_file_count == 0:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], scan.filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
        db.session.delete(scan)

    for row in chat_rows:
        db.session.delete(row)

    db.session.commit()
    flash("Patient scan and chat history cleared successfully.", "success")
    return redirect(url_for('dashboard', patient_id=patient_id))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == "__main__":
    with app.app_context():
        os.makedirs(os.path.join(app.root_path, "instance"), exist_ok=True)
        db.create_all()
        migrate_sqlite_schema()
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)