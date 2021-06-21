from flask import Flask, make_response

app = Flask('Genre Detection API')

with app.app_context():
    import movie_classifier.api.health
    import movie_classifier.api.predict
    import movie_classifier.api.training


@app.route('/', methods=['GET'])
def welcome():

    return make_response({'Welcome to Movie Detection API': 'Ok'})