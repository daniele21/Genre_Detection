from flask import make_response, jsonify, request
from flask import current_app as app

import logging

from movie_classifier.constants.paths import MODELS_DIR
from movie_classifier.scripts.model.inference import genre_prediction
from movie_classifier.scripts.pipeline.dataset_generation import generate_test_sample
from movie_classifier.scripts.utils.loadings import load_latest_model, load_tokenizer

logger = logging.getLogger('Prediction API')


@app.route('/predict', methods=['POST'])
def predict():

    params = request.json

    title = params['title']
    description = params['description']
    model_dir = params['model_dir'] if 'model_dir' in params else MODELS_DIR

    # Loading model
    model, model_dir = load_latest_model(model_dir)

    # Loading Tokenizer
    tokenizer = load_tokenizer(model_dir)

    # Generate dataset for the model processing
    descr_sample = generate_test_sample(description, tokenizer)

    # Model inference
    genre = genre_prediction(model, descr_sample, tokenizer)

    response = {'title': title,
                'description': description,
                'genre': genre}

    return make_response(jsonify(response))
