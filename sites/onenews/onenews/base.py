from flask import Flask, render_template

from onenews.db import get_db
from onenews.auth import get_current_user

app = Flask(__name__)

@app.route('/home')
def home():
    db = get_db()

    user = get_current_user()

    if not user:
        return render_template('auth/login.html')

    return 'Hello, World!'