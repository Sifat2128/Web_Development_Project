"""Web-layer safety checks. Run: python -m unittest test_app"""
import io
import os
import tempfile
import unittest

os.environ["SECRET_KEY"] = "test-key"
os.environ["DATA_DIR"] = tempfile.mkdtemp()
os.environ["MAIL_USERNAME"] = "noreply@test.local"

from app import app, db, mail, limiter, User, Patient, Scan, PendingSignup  # noqa: E402

app.config.update(TESTING=True, SESSION_COOKIE_SECURE=False)
app.extensions["mail"].suppress = True
limiter.enabled = False


def png_bytes():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (64, 64), 128).save(buf, "PNG")
    buf.seek(0)
    return buf


class WebTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        with app.app_context():
            db.drop_all()
            db.create_all()
            a, b = User(email="a@x.com", password="x"), User(email="b@x.com", password="x")
            db.session.add_all([a, b])
            db.session.commit()
            pa = Patient(name="Alice", age=40, user_id=a.id)
            pb = Patient(name="Bob", age=50, user_id=b.id)
            db.session.add_all([pa, pb])
            db.session.commit()
            self.a_id, self.pa_id, self.pb_id = a.id, pa.id, pb.id

    def login_as_a(self):
        with self.client.session_transaction() as s:
            s["user_id"] = self.a_id

    def test_chat_about_other_doctors_patient_is_404(self):
        self.login_as_a()
        r = self.client.post(f"/api/chat/{self.pb_id}", json={"message": "name?"})
        self.assertEqual(r.status_code, 404)

    def test_upload_rejects_html_and_saves_png_outside_static(self):
        self.login_as_a()
        self.client.post(f"/dashboard/{self.pa_id}", data={"file": (io.BytesIO(b"<script>"), "evil.html")})
        self.client.post(f"/dashboard/{self.pa_id}", data={"file": (png_bytes(), "scan.png")})
        self.client.post(f"/dashboard/{self.pa_id}", data={"file": (png_bytes(), "scan.png")})
        with app.app_context():
            names = [s.filename for s in Scan.query.all()]
        self.assertEqual(len(names), 2)                      # html rejected
        self.assertEqual(len(set(names)), 2)                 # same name did not overwrite
        self.assertTrue(all(n.endswith("_scan.png") for n in names))
        for n in names:
            self.assertTrue(os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], n)))
        self.assertNotIn("static", app.config["UPLOAD_FOLDER"])

        self.client.post(f"/clear_history/{self.pa_id}")
        for n in names:
            self.assertFalse(os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], n)))

    def test_unreadable_image_is_not_labelled_normal(self):
        self.login_as_a()
        r = self.client.post(f"/dashboard/{self.pa_id}", data={"file": (io.BytesIO(b"not an image"), "fake.png")},
                             follow_redirects=True)
        self.assertIn(b"Could not analyse this image", r.data)
        with app.app_context():
            self.assertEqual(Scan.query.count(), 0)          # no fake "Normal" result saved
        self.assertFalse([f for f in os.listdir(app.config["UPLOAD_FOLDER"]) if f.endswith("_fake.png")])  # bad file removed

    def test_dashboard_shows_no_random_accuracy(self):
        self.login_as_a()
        self.client.post(f"/dashboard/{self.pa_id}", data={"file": (png_bytes(), "scan.png")})
        r = self.client.get(f"/dashboard/{self.pa_id}")
        self.assertIn(b"Model Test Accuracy", r.data)
        with app.app_context():
            self.assertEqual(Scan.query.one().accuracy, 0.0)

    def test_patient_form_rejects_bad_age(self):
        self.login_as_a()
        self.client.post("/patient_info", data={"name": "Eve", "age": "abc"})
        with app.app_context():
            self.assertIsNone(Patient.query.filter_by(name="Eve").first())

    def test_signup_code_stays_server_side_and_locks_after_5_wrong_tries(self):
        with mail.record_messages() as outbox:
            self.client.post("/signup", data={"email": "new@x.com", "password": "pw"})
        code = outbox[0].body.split(": ")[1][:6]
        with self.client.session_transaction() as s:
            self.assertNotIn(code, str(dict(s)))             # cookie never holds the code

        wrong = "000000" if code != "000000" else "111111"
        for _ in range(5):
            self.client.post("/verify_signup", data={"otp": wrong})
        self.client.post("/verify_signup", data={"otp": code})  # right code, but locked now
        with app.app_context():
            self.assertIsNone(User.query.filter_by(email="new@x.com").first())

    def test_signup_with_correct_code_creates_user(self):
        with mail.record_messages() as outbox:
            self.client.post("/signup", data={"email": "ok@x.com", "password": "pw"})
        code = outbox[0].body.split(": ")[1][:6]
        self.client.post("/verify_signup", data={"otp": code})
        with app.app_context():
            self.assertIsNotNone(User.query.filter_by(email="ok@x.com").first())
            self.assertIsNone(PendingSignup.query.filter_by(email="ok@x.com").first())


if __name__ == "__main__":
    unittest.main()
