from flask import Flask, render_template, request, send_file, redirect, flash
from werkzeug.utils import secure_filename
import os
from Image_processing import convert_fits_to_image

app = Flask(__name__)

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


@app.route('/project', methods=['GET', 'POST'])
def project():
    if request.method == 'POST':
        # Handle POST request
        projname = request.form['projname']
        dataset = request.files['dataset']

        # Save dataset file to disk (optional)
        dataset.save('uploaded_dataset.fits')

        #  Process the uploaded dataset
        output_image_filename = 'processed_image.png'
        convert_fits_to_image('uploaded_dataset.fits', output_image_filename)

        # Return the processed image file to the client
        return send_file(output_image_filename, mimetype='image/png')
    else:
        # Handle GET request
        return render_template('project.html')

if __name__ == '__main__':
    app.run(debug=True)
