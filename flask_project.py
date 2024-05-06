from flask import Flask, render_template, request, send_file, redirect, flash , url_for
from werkzeug.utils import secure_filename
import os
from Image_processing import convert_fits_to_image
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# MySQL Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Happylola.123@localhost/espacio'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define your Project model
class Project(db.Model):
    project_id = db.Column(db.Integer, primary_key=True)
    projectname = db.Column(db.String(255))
    source = db.Column(db.String(255))
    files = db.Column(db.String(255))

# Set the secret key for session management and flashing messages
app.secret_key = '0'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/signup')
def signup():
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
    return render_template('reports.html')

@app.route('/messages')
def messages():
    return render_template('messages.html')

# Define the upload folder path
UPLOAD_FOLDER = 'uploads'

# Set the upload folder configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/project', methods=['GET', 'POST'])
def project():
    if request.method == 'POST':
            projname = request.form['projname']
            source = request.form['source']
            files = request.files['dataset']

            # Save dataset file to disk
            filename = secure_filename(files.filename)
            files.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Insert project data into the database
            new_project = Project(projectname=projname, source=source, files=filename)
            db.session.add(new_project)
            db.session.commit()

            # Process the uploaded dataset (optional)

            # Return success message or redirect to another page
            flash('Project created successfully', 'success')
            return redirect(url_for('allreports'))
    else:
            return render_template('project.html')

if __name__ == '__main__':
    app.run(debug=True)
