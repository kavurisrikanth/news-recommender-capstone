import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash

import onenews
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

    if onenews.ANALYSIS_DATA is None:
        return 'Analysis data not found!'

    data = onenews.ANALYSIS_DATA
    top_10_articles = data.get_top_10_articles_for_user(user_id)

    a = top_10_articles.to_dict('records')
    # print(a)
    return render_template('home.html', posts=a)

@bp.route('/article/<int:id>', methods=('GET',))
def get_article(id):
    user_id = None
    if 'user_id' in session:
        user_id = session['user_id']

    if not user_id:
        return redirect(url_for('auth.login'))

    article = onenews.ANALYSIS_DATA.get_article_by_id(id)

    if article is None:
        return 'Article not found!'

    more = onenews.ANALYSIS_DATA.get_more_articles_for_user(id, user_id).to_dict('records')
    print(more)

    return render_template('article.html', article=article, more=more)