from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# Hardcoded user credentials for demonstration
USER_CREDENTIALS = {
    'email': 'user@example.com',
    'password': '12345'
}

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/')
def signupindex():
    return render_template('signup.html')

@app.route('/signup', methods=['POST'])
def signup():
    # Process form submission here
    # Example: Access form data and perform backend tasks
    fn = request.form['fn']
    ln = request.form['ln']
    age = request.form['age']
    email = request.form['email']
    password = request.form['password']
    cpass = request.form['cpass']
    affiliation = request.form['affiliation']
    country = request.form['country']

    # Example: Print form data (you can replace this with your actual backend logic)
    print(f'First Name: {fn}')
    print(f'Last Name: {ln}')
    print(f'Age: {age}')
    print(f'Email: {email}')
    print(f'Password: {password}')
    print(f'Confirm Password: {cpass}')
    print(f'Affiliation: {affiliation}')
    print(f'Country: {country}')

    # Redirect to home page after successful sign-up
    return redirect('/')

@app.route('/signinindex')
def signinindex():
    return render_template('signin.html')

@app.route('/signin', methods=['POST'])
def signin():
    # Get form data
    email = request.form['email']
    password = request.form['password']

    # Check if credentials match
    if email == USER_CREDENTIALS['email'] and password == USER_CREDENTIALS['password']:
        # Redirect to home page after successful sign-in
        return redirect('/homeindex')
    else:
        # If credentials don't match, redirect back to sign-in page with an error message
        return render_template('signin.html', error='Invalid email or password')

@app.route('/homeindex')
def homeindex():
    return render_template('index.html')

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

@app.route('/document')
def document():
    return render_template('document.html')

@app.route('/editdeleteusers')
def editdeleteusers():
    return render_template('editdeleteusers.html')


@app.route('/fullreport')
def fullreport():
    return render_template('fullreport.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/reports')
def reports():
    return render_template('reports.html')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=9000)
