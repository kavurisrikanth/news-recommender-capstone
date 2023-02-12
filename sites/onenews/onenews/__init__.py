import os
import threading

from flask import Flask, render_template, g, url_for

from onenews.analysis import get_data

ANALYSIS_DATA = None

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'onenews.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from . import db
    db.init_app(app)

    from . import auth
    app.register_blueprint(auth.bp)

    from . import home
    app.register_blueprint(home.bp)

    perform_analysis(app, os.path.join(app.root_path, 'data'))

    return app

def perform_analysis(app, path):
    print('*** Performing analysis')
    global ANALYSIS_DATA
    ANALYSIS_DATA = get_data(path)
    print('*** Data ready')
    print(f'*** Num users: {ANALYSIS_DATA.n_users}')
