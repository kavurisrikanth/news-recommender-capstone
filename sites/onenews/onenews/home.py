import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash

from onenews.db import get_db
from onenews.auth import get_current_user

bp = Blueprint('home', __name__)

@bp.route('/', methods=('GET', 'POST'))
def home():
    db = get_db()

    user_id = None
    if 'user_id' in session:
        user_id = session['user_id']

    print(f'*** user_id: {user_id}')

    if not user_id:
        return redirect(url_for('auth.login'))

    return render_template('home.html')
