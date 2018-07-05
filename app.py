from waitress import serve
from flask import Flask
from apis import blueprint as api

app = Flask(__name__)
app.register_blueprint(api, url_prefix='/api/v1')

serve(app)
