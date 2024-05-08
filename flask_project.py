from flask import Flask, render_template, request, redirect, flash, session, url_for
from werkzeug.utils import secure_filename
import os
from flask_sqlalchemy import SQLAlchemy
import base64
import uuid
import io
import zipfile
import threading
from uuid import uuid4
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# MySQL Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Happylola.123@localhost/espacio'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define your AccountInfo model
class AccountInfo(db.Model):
    acc_id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    age = db.Column(db.Integer)
    username = db.Column(db.String(10), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    acc_password = db.Column(db.String(250), nullable=False)

# Define your Project model
class Project(db.Model):
    project_id = db.Column(db.String(36), primary_key=True, default=str(uuid.uuid4()))
    projectname = db.Column(db.String(255))
    source = db.Column(db.String(255))
    files = db.Column(db.LargeBinary)
    acc_id = db.Column(db.Integer, db.ForeignKey('account_info.acc_id'))

# Set the secret key for session management and flashing messages
app.secret_key = '0'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = AccountInfo.query.filter_by(email=email).first()
        if user and check_password_hash(user.acc_password, password):
            session['acc_id'] = user.acc_id
            flash('Logged in successfully!', 'success')
            return redirect(url_for('project'))
        else:
            flash('Invalid username or password. Please try again.', 'error')
    return render_template('signin.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first_name = request.form['fn']
        last_name = request.form['ln']
        age = request.form['age']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        hashed_password = generate_password_hash(password)

        new_user = AccountInfo(first_name=first_name, last_name=last_name, age=age,
                               username=username, email=email, acc_password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Your account has been created! You can now sign in.', 'success')
        return redirect(url_for('signin'))
    return render_template('signup.html')

@app.route('/document')
def documentation():
    return render_template('document.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/account')
def account():
    return render_template('account.html')

@app.route('/admindashboard')
def admindashboard():
    return render_template('admindashboard.html')

@app.route('/allreports')
def allreports():
    return render_template('allreports.html')

@app.route('/contactus')
def contactus():
    return render_template('contactus.html')

@app.route('/editdeleteusers')
def editdeleteusers():
    return render_template('editdeleteusers.html')

@app.route('/fullreport')
def fullreport():
    return render_template('fullreport.html')

@app.route('/reports')
def reports():
    if 'acc_id' in session:
        # Fetch projects associated with the logged-in user
        acc_id = session['acc_id']
        projects = Project.query.filter_by(acc_id=acc_id).all()
        return render_template('reports.html', projects=projects)
    else:
        # If no user is logged in, redirect to the login page
        flash('Please log in to view reports', 'error')
        return redirect(url_for('signin'))


@app.route('/messages')
def messages():
    return render_template('messages.html')

# def process_files_async(files, projname, source, acc_id):
    
@app.route('/project', methods=['GET', 'POST'])
def project():
    if request.method == 'POST':
        if 'acc_id' in session:  # Check if user is logged in
            projname = request.form['projname']
            source = request.form['source']
            files = request.files.getlist('dataset')  # Use getlist for multiple files
            acc_id = session['acc_id']
            # Function to process files asynchronously
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                for file in files:
                    file_content = file.read()
                    file_name = secure_filename(file.filename)
                    zip_file.writestr(file_name, file_content)

            zip_contents = zip_buffer.getvalue()
            
            new_project = Project(project_id=str(uuid4()), projectname=projname, source=source, files=zip_contents, acc_id=acc_id)
            db.session.add(new_project)
            db.session.commit()

            # Execute functions based on checked checkboxes
            if 'detect' in request.form:
                # Execute detection-related function
                # Example: detect_function()
                pass
            
            if 'track' in request.form:
                # Execute tracking-related function
                # Example: track_function()
                pass
            
            if 'collision' in request.form:
                # Execute collision prediction-related function
                # Example: collision_function()
                pass

            # Flash success message
            flash('Project created successfully', 'success')
            return redirect(url_for('reports'))
        else:
            flash('Please log in to create a project', 'error')
            return redirect(url_for('signin'))  # Redirect to login page if user is not logged in
    else:
        return render_template('project.html')
if __name__ == '__main__':
    app.run(debug=True)
