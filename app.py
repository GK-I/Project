from flask import Flask, redirect,render_template
from flask import request
from flask import url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin,LoginManager,login_user,logout_user,login_required,current_user,logout_user
from werkzeug.security import generate_password_hash,check_password_hash
from flask import flash

app=Flask(__name__)
app.secret_key='MITS@123'

app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///teachers.db'
db=SQLAlchemy(app)


login_manager=LoginManager()
login_manager.login_view='login'
login_manager.init_app(app)

class Teacher(UserMixin,db.Model):
    id=db.Column(db.Integer, primary_key=True,autoincrement=True)
    name=db.Column(db.String(100), nullable=False)
    email=db.Column(db.String(100), unique=True,nullable=False)
    password=db.Column(db.String(100),nullable=False)
app.app_context().push()
db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return Teacher.query.get(int(user_id))

@app.route('/')
def hello():
    return redirect(url_for('login'))

@app.route('/test-insert')
def test_insert():
    teacher = Teacher(name="Test Teacher", email="test@eduvision.com", password=generate_password_hash("password", method='pbkdf2:sha256'))
    db.session.add(teacher)
    db.session.commit()
    return "Inserted test record!"

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method=='POST':
        email=request.form['email']
        password=request.form['password']
        user=Teacher.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again')
            return redirect(url_for('login'))
        login_user(user)
        return redirect(url_for('profile'))
    return render_template("login.html")

'''@app.route('/signup',methods=['POST','GET'])
def signup():
    if request.method=='GET':
        return render_template("signup.html")
    name=request.form['name']
    email=request.form['email']
    password=request.form['password']
    user=Teacher.query.filter_by(email=email).first()
    if user:
        flash('Email already exists')
        return redirect(url_for('signup'))
    new_user=Teacher(name=name, email=email, password=generate_password_hash(password, method='pbkdf2:sha256'))
    db.session.add(new_user)
    db.session.commit()
    flash('Account created successfully')
    return redirect(url_for('login'))'''

@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'GET':
        return render_template("signup.html")
    
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    
    print(f"Received Name: {name}, Email: {email}")  # Debugging line to check if form data is correct
    
    user = Teacher.query.filter_by(email=email).first()
    if user:
        flash('Email already exists')
        return redirect(url_for('signup'))
    
    new_user = Teacher(name=name, email=email, password=generate_password_hash(password, method='pbkdf2:sha256'))
    db.session.add(new_user)
    db.session.commit()
    
    print("User added to DB")  # Debugging line to confirm if user is added to the database
    flash('Account created successfully')
    return redirect(url_for('login'))


@app.route('/profile')
@login_required
def profile():
    return render_template("home.html", user=current_user)
@app.route('/logout')

@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.after_request
def add_header(response):
    if request.endpoint not in ['static']:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response
app.run(debug=True)
