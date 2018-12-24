
from flask import Flask
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

from deep_learning import DeepLearning

# tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dist')
# static_folder = "dist"
# app = Flask(__name__,static_folder=static_folder, template_folder=tmpl_dir)
app = Flask(__name__)
app.debug = False


@app.route("/")
def hello():
    return "Hello World!"


if __name__ == '__main__':

    # mqttClient = MQTTClient()
    # mqttClient.connect()

    deepLearning = DeepLearning()

    server = pywsgi.WSGIServer(('0.0.0.0', 8083), app, handler_class=WebSocketHandler)
    server.serve_forever()