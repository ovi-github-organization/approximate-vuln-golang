from flask import Flask, request, make_response

app = Flask(__name__)

@app.before_request
def prepare_response():
    response = make_response()
    #response.headers['Vary'] = 'Cookie'

@app.route('/')
def index():
    # Your view logic here
    return 'Hello, world!'

if __name__ == '__main__':
    app.run()
