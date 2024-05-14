#from flask import request, render_template
#from flask_wtf import FlaskForm
#from wtforms import StringField, PasswordField
#from wtforms.validators import InputRequired
#
#class LoginForm(FlaskForm):
#    username = StringField('Username', validators=[InputRequired()])
#    password = PasswordField('Password', validators=[InputRequired()])
#
#from form import LoginForm
#
#@app.route('/login', methods=['POST'])
#def login():
#    form = LoginForm(request.form)
#    if form.validate():
#        username = form.username.data
#        password = form.password.data
#        # Add your authentication logic here
#    else:
#        # If validation fails, send the errors to the template
#        return render_template('login.html', errors=form.errors)
#    
#{% for field, errors in errors.items() %}
#    {% for error in errors %}
#        <div class="error">{{ error }}</div>
#    {% endfor %}
#{% endfor %}