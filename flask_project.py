from flask import Flask, render_template, request, redirect, flash, session, url_for, jsonify
from flask import send_file, make_response
from werkzeug.utils import secure_filename
from sqlalchemy import func
import os
from flask_sqlalchemy import SQLAlchemy
import base64
import uuid
import io
import zipfile
import tempfile
import time
import shutil
from sqlalchemy.orm import relationship
import pandas as pd
from flask_mail import Mail, Message
from zipfile import ZipFile
import threading
import cv2
from uuid import uuid4
from werkzeug.security import generate_password_hash, check_password_hash
from OOP.Detection.Classification import DebrisAnalyzer  
from OOP.Detection.conversion import convert_fits_to_image
from OOP.Detection.images_Preprocessing.Otsu_Thresholding import otsu_thresholding_folder 
from OOP.Detection.images_Preprocessing.iterative_Threshholding import iterative_thresholding_folder
from OOP.Detection.object_labeling import detect_objects
from OOP.Tracking.Images_to_Vid import images_to_video
from OOP.Tracking.optical_flow_fernback import OpticalFlowAnalyzer

app = Flask(__name__)

# MySQL Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Happylola.123@localhost/espacio'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# # MySQL Configuration
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:salmabaligh123@localhost/espacio'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

# Define your AccountInfo model
class AccountInfo(db.Model):
    acc_id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    age = db.Column(db.Integer)
    username = db.Column(db.String(10), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    acc_password = db.Column(db.String(250), nullable=False)
    project_count = db.Column(db.Integer, default=0)
    # Relationship to projects
    projects = relationship('Project', backref='account', lazy=True)
    # Relationship to contact messages
    contact_messages = relationship('ContactMessage', backref='account', lazy=True, cascade='all, delete-orphan')


# Define your Project model
class Project(db.Model):
    project_id = db.Column(db.String(36), primary_key=True, default=str(uuid.uuid4()))
    projectname = db.Column(db.String(255))
    source = db.Column(db.String(255))
    files = db.Column(db.LargeBinary)
    detection = db.Column(db.LargeBinary) 
    tracking = db.Column(db.LargeBinary)
    acc_id = db.Column(db.Integer, db.ForeignKey('account_info.acc_id'))

# Define your SQLAlchemy model for the contact messages
class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    category = db.Column(db.String(50))
    priority = db.Column(db.String(20))
    copy = db.Column(db.Boolean)
    message = db.Column(db.Text)
    acc_id = db.Column(db.Integer, db.ForeignKey('account_info.acc_id', ondelete='CASCADE'))


# Set the secret key for session management and flashing messages
app.secret_key = '0'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email'].strip()  # Trim whitespace
        password = request.form['password']

        # Define the specific admin email and password
        admin_email = "admin@gmail.com"
        admin_password = "1234"

        if email.lower() == admin_email.lower() and password == admin_password:
            session['admin'] = True
            flash('Logged in as admin!', 'success')
            return redirect(url_for('admindashboard'))

        user = AccountInfo.query.filter(func.lower(AccountInfo.email) == func.lower(email)).first()  # Case-insensitive email comparison
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


from flask import request, redirect, url_for

@app.route('/admin_users', methods=['GET', 'POST'])
def admin_users():
    if 'admin' in session:
        if request.method == 'POST':
            acc_id = request.form.get('acc_id')
            if acc_id:
                user = AccountInfo.query.filter_by(acc_id=acc_id).first()
                if user:
                    db.session.delete(user)
                    db.session.commit()
                    flash(f'User {user.first_name} {user.last_name} deleted successfully', 'success')
                    return redirect(url_for('admin_users'))
                else:
                    flash('User not found', 'error')
                    return redirect(url_for('admin_users'))

        users = AccountInfo.query.all()  # Fetch all user data from the database
        return render_template('admin_users.html', users=users)
    else:
        flash('Unauthorized access!', 'error')
        return redirect(url_for('signin'))


@app.route('/account')
def account():
    if 'acc_id' in session:
        # Fetch the account information for the current user
        acc_id = session['acc_id']
        user = AccountInfo.query.filter_by(acc_id=acc_id).first()
        
        if user:
            # Pass the user information to the template
            return render_template('account.html', user=user)
        else:
            flash('Account information not found', 'error')
            return redirect(url_for('signin'))
    else:
        flash('Please log in to view your account', 'error')
        return redirect(url_for('signin'))

@app.route('/signout')
def signout():
    # Clear the session data
    session.clear()
    # Redirect to the home page or any other desired page
    return redirect(url_for('index'))

@app.route('/admindashboard')
def admindashboard():
    return render_template('admindashboard.html')

@app.route('/allprojects', methods=['GET', 'POST'])
def allprojects():
    if request.method == 'POST':
        project_id = request.form.get('project_id')
        if project_id:
            project = Project.query.get(project_id)
            if project:
                db.session.delete(project)
                db.session.commit()
                return redirect(url_for('allprojects'))

    projects = db.session.query(Project).join(AccountInfo).all()
    return render_template('allprojects.html', projects=projects)



@app.route('/contactus', methods=['GET', 'POST'])
def contactus():
    if request.method == 'POST':
        # Ensure user is logged in and has an acc_id in session
        if 'acc_id' in session:
            acc_id = session['acc_id']
        else:
            return jsonify({'error': 'User not logged in'}), 401  # Unauthorized

        # Process form submission
        name = request.form.get('name', '')
        email = request.form.get('email', '')
        category = request.form.get('category', '')
        priority = request.form.get('priority', '')
        copy = 'copy' in request.form
        message = request.form.get('message', '')

        # Save to database
        new_message = ContactMessage(name=name, email=email, category=category, priority=priority, copy=copy, message=message, acc_id=acc_id)
        db.session.add(new_message)
        db.session.commit()

        # Return success message
        return jsonify({'message-popup': 'Message sent successfully'})

    return render_template('contactus.html')

@app.route('/fullreport')
def fullreport():
    # Assuming your DebrisAnalyzer setup and object_data retrieval here
    analyzer = DebrisAnalyzer(threshed_directory="OOP\\2024-001", csv_file_path="output.csv")
    object_data = analyzer.process_images()
    total_objects = analyzer.get_total_objects(object_data)

    # Render the HTML template and pass total_objects to it
    return render_template('full_reports.html', total_objects=total_objects)
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

# @app.route('/detection_output')
# def detection():
#     return render_template('detection_output.html')
# Initialize Flask-Mail
mail = Mail(app)
@app.route('/messages')
def messages():
    messages = ContactMessage.query.all()
    return render_template('messages.html', messages=messages)

@app.route('/reply_message/<int:message_id>', methods=['POST'])
def reply_message(message_id):
    reply_message = request.form.get('reply_message','')
    message = ContactMessage.query.get(message_id)
    
    if message:
        # Send email to the user (replace with actual email sending code)
        # Example using Flask-Mail:
        # msg = Message('Reply from Admin', sender='admin@example.com', recipients=[message.email])
        # msg.body = reply_message
        # mail.send(msg)
        
        # # Update message in database if needed
        # message.reply = reply_message
        # db.session.commit()
        
        return jsonify({'message': f'Reply sent to {message.name} at {message.email}.'})
    else:
        return jsonify({'message': 'Message not found.'}), 404

@app.route('/delete_message/<int:message_id>', methods=['POST'])
def delete_message(message_id):
    message = ContactMessage.query.get(message_id)
    
    if message:
        db.session.delete(message)
        db.session.commit()
        return jsonify({'message': f'Message from {message.name} deleted successfully.'})
    else:
        return jsonify({'message': 'Message not found.'}), 404
# def process_files_async(files, projname, source, acc_id):
# Function to process files asynchronously
            # zip_buffer = io.BytesIO()
            
            # with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            #     for file in files:
            #         file_content = file.read()
            #         file_name = secure_filename(file.filename)
            #         zip_file.writestr(file_name, file_content)

            # zip_contents = zip_buffer.getvalue()
            
            # new_project = Project(project_id=str(uuid4()), projectname=projname, source=source, files=zip_contents, acc_id=acc_id)
            # db.session.add(new_project)
            # db.session.commit()


from flask import send_file

@app.route('/download_video/<project_id>')
def download_video(project_id):
    # Fetch the project by its ID
    project = Project.query.get(project_id)
    if project:
        # Create a response containing the video data
        response = make_response(project.tracking)
        # Set the Content-Disposition header to specify the filename
        response.headers['Content-Disposition'] = f'attachment; filename=project_video.mp4'
        # Set the content type
        response.headers['Content-Type'] = 'video/mp4'
        return response  # Return the response object
    else:
        flash('Project not found', 'error')
        return redirect(url_for('reports'))




@app.route('/view_csv/<project_id>')
def view_csv(project_id):
    # Fetch the project by its ID
    project = Project.query.get(project_id)
    if project:
        # Create a response containing the CSV file content
        response = make_response(project.detection)
        # Set the Content-Disposition header to specify the filename
        response.headers['Content-Disposition'] = f'attachment; filename=output.csv'
        # Set the content type
        response.headers['Content-Type'] = 'text/csv'
        return response
    else:
        flash('Project not found', 'error')
        return redirect(url_for('reports'))

  # Add this import statement at the top of your file

def all_files_uploaded(files):
    # Check if all files are uploaded
    for file in files:
        if not file:  # Check if file is None
            return None
    # Create a temporary directory to store files
    temp_dir = "fits_directory"
    os.makedirs(temp_dir, exist_ok=True)
    # Save uploaded files in the temporary directory
    for file in files:
        file_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(file_path)
    return temp_dir



@app.route('/project', methods=['GET', 'POST'])
def project():
    if request.method == 'POST':
        if 'acc_id' in session:  # Check if user is logged in
            projname = request.form['projname']
            source = request.form['source']
            files = request.files.getlist('dataset')  # Use getlist for multiple files
            acc_id = session['acc_id']
            detection = 'detection' in request.form
            tracking = 'tracking' in request.form
            collision = 'collision' in request.form

            # Store project details in session for later use on reports page
            session['project_details'] = {
            'projname': projname,
            'source': source,
            'detection': detection,
            'tracking': tracking,
            'collision': collision
        }
        
            

            # Ensure all files are uploaded before proceeding
            temp_dir = all_files_uploaded(files)

            if not temp_dir:
                flash('No FITS files uploaded', 'error')
                return redirect(url_for('project'))

           

            try:
                # Process FITS files
                csv_content = process_fits_files(temp_dir)
                

                

                #  # Check if detection is selected
                # if request.form.get('detect'):
                #     for filename in os.listdir('iterat_images'):
                #         # Load the binary image
                #         binary_image = cv2.imread(os.path.join('iterat_images', filename), cv2.IMREAD_GRAYSCALE)

                #         # Detect objects in the binary image
                #         detected_objects, annotated_image = detect_objects(binary_image)

                #         # Save the annotated image to the output folder
                #         output_path = os.path.join('annotated_images', filename)
                #         cv2.imwrite(output_path, annotated_image)
                #         return render_template('detection_output.html', detected_objects=detected_objects,
                #                                 annotated_image=annotated_image)

                # Check if tracking is selected
                if request.form.get('track'):
                    
                    images_to_video('iterat_images', 'OG.MP4', 5)

                    
                    analyzer = OpticalFlowAnalyzer('OG.MP4', 'fernbackOUT.MP4')
                    analyzer.process_video('track.csv')

                    # Read the generated video file as binary data
                    with open('fernbackOUT.MP4', 'rb') as video_file:
                        video_data = video_file.read()
                
                else:
                    video_data = None  # Ensure video_data is initialized


                new_project = Project(project_id=str(uuid.uuid4()), projectname=projname, source=source, detection=csv_content, tracking=video_data, acc_id=acc_id)
                db.session.add(new_project)
                db.session.commit()

                # Increment the project count for the user if user exists
                user = AccountInfo.query.filter_by(acc_id=acc_id).first()
                if user:
                    user.project_count = user.project_count + 1 if user.project_count is not None else 1
                    db.session.commit()
               
                flash('Project created successfully', 'success')
                return redirect(url_for('reports'))

            finally:
                # Delete temporary directory and its contents
                # shutil.rmtree(temp_dir)
                ("ay haga")
        else:
            flash('Please log in to create a project', 'error')
            return redirect(url_for('signin'))  # Redirect to login page if user is not logged in
    else:
        return render_template('project.html')


def process_fits_files(temp_dir):
    # fits_directory = 'fits_directory'
    # os.makedirs(temp_dir, exist_ok=True)

    # # Save FITS files in the fits_directory
    # for file in temp_dir:
    #     file_path = os.path.join(fits_directory, secure_filename(file.filename))
    #     file.save(file_path)

    # Create directories for images preprocessing
    images_directory = 'images_directory'
    os.makedirs(images_directory, exist_ok=True)

    otsu_images = 'otsu_images'
    os.makedirs(otsu_images, exist_ok=True)

    iterat_images = 'iterat_images'
    os.makedirs(iterat_images, exist_ok=True)

    # Perform image conversion and preprocessing
    convert_fits_to_image(temp_dir, images_directory)
    otsu_thresholding_folder(images_directory, otsu_images)
    iterative_thresholding_folder(images_directory, iterat_images)

    # Perform debris analysis
    csv_file_path = os.path.join('output.csv')
    analyzer = DebrisAnalyzer(iterat_images, csv_file_path)
    analyzer.process_images()

    # Check if the CSV file exists
    if os.path.exists(csv_file_path):
        # Read the content of the CSV file
        with open(csv_file_path, 'rb') as csv_file:
            csv_content = csv_file.read()


        return csv_content
    else:
        # If the CSV file doesn't exist, return None
        flash('Error: CSV file not generated', 'error')
        return None

if __name__ == '__main__':
    app.run(debug=True)