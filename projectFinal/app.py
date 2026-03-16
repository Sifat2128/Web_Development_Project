from flask import Flask, render_template,request,redirect,url_for,session,flash
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash , check_password_hash
import random
import os
from sqlalchemy import func
from datetime import timedelta
app=Flask(__name__)
app.secret_key="pulmoscan_secret"

app.config['SQLALCHEMY_DATABASE_URI']='mysql+pymysql://root:@localhost/pulmoscan'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
db=SQLAlchemy(app)

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT']=587
app.config['MAIL_USE_TLS']=True
app.config['MAIL_USERNAME']='farzanafaizaborno2023@gmail.com'
app.config['MAIL_PASSWORD']='lzfpgnqhwxrdtemp'
mail=Mail(app)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

class User(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    email=db.Column(db.String(150),unique=True,nullable=False)
    password=db.Column(db.String(255),nullable=False)
    scans=db.relationship('Scan',backref='patient',lazy=True)
class Scan(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    filename=db.Column(db.String(255),nullable=False)
    result=db.Column(db.String(50),nullable=False)
    accuracy=db.Column(db.Float , nullable=False)
    user_id=db.Column(db.Integer,db.ForeignKey('user.id'),nullable=False)
    
with app.app_context():
    db.create_all()
@app.route("/")
def home():
    return render_template("homepage.html")
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method=="POST":
        email=request.form.get("email")
        password=request.form.get("password")
        user= User.query.filter_by(email=email).first()
        if user:
            flash('This email is already registered. Please log in here.','info')
            return redirect(url_for('login'))
        otp= str(random.randint(100000,999999))
        session['temp_user']={
            'email':email,
            'password': generate_password_hash(password),
            'otp': otp
        }
        try:
            msg=Message('Verify your PulmoScan Account',
                        sender=app.config['MAIL_USERNAME'],
                        recipients=[email])
            msg.body=f"Welcome to PulmoScan!Your Verification code is:{otp}"
            mail.send(msg)
            return redirect(url_for('verify_signup'))
        except Exception as e:
            flash('Error sending mail.Please check your configuration.','error')
            print(f"Mail Error:{e}")
    return render_template('signup.html')
@app.route('/resend_otp')
def resend_otp():
    temp_data = session.get('temp_user')
    if not temp_data:
        flash('Session expired or invalid request. Please sign up again.','error')
        return redirect(url_for('signup'))
    new_otp=str(random.randint(100000,999999))
    temp_data['otp']=new_otp
    session['temp_user']=temp_data
    session.modified=True
    try: 
        msg=Message('New Verification Code-PulmoScan',
                    sender=app.config['MAIL_USERNAME'],
                    recipients=[temp_data['email']])
        msg.body=f"Your new PulmoScan verification code is:{new_otp}"
        mail.send(msg)
        flash('A new verification code has been sent to your email.','success')
    except Exception as e:
        flash('Error sending mail. Please try again later.','error')
        print(f"Mail error:{e}")
    return redirect(url_for('verify_signup'))
@app.route('/verify',methods=['GET','POST'])
def verify_signup():
    if request.method == 'POST':
        entered_otp=request.form.get('otp')
        temp_data=session.get('temp_user')
        
        if temp_data and entered_otp == temp_data['otp']:
            new_user=User(email=temp_data['email'],password=temp_data['password'])
            db.session.add(new_user)
            db.session.commit()
            session.pop('temp_user',None)
            flash('Registration successful! Please login','success')
            return redirect(url_for('login'))
        else:
            flash('Invalid verification code.','error')
    return render_template('verify_signup.html')
@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST':
        email=request.form.get('email')
        password=request.form.get('password')
        remember=request.form.get('remember')
        user=User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password,password):
            session['user_id']=user.id 
            session['user_email']=user.email 
            if remember:
                session.permanent=True 
                app.permanent_session_lifetime=timedelta(days=30)
            else:
                session.permanent=False
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again.','error')
    return render_template('login.html')
@app.route("/dashboard", methods=['GET','POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    u_id=session['user_id']
    u_name=session['user_email'].split('@')[0].capitalize()
    scan_count = Scan.query.filter_by(user_id=u_id).count()
    avg_accuracy = db.session.query(func.avg(Scan.accuracy)).filter(Scan.user_id == u_id).scalar() or 0
    return render_template("dashboard.html",
                           user_name=u_name,
                           scan_count=scan_count,
                           avg_accuracy=round(avg_accuracy,1))
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))
if __name__ =="__main__":
    app.run(debug=True)


