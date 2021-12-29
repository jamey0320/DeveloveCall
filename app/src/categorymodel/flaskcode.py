import dlmodel
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
        return 'Hello, World!'

@app.route('/<param>', methods=['POST'])
def hello_user(param):
        result = dlmodel.main(param)
        return '%s'% result

if __name__ == "__main__":
        app.run(host='0.0.0.0', port=5000, debug=True)
