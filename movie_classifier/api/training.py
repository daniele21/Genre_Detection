from flask import make_response, jsonify, request
from flask import current_app as app

import logging

from movie_classifier.constants.params import DEFAULT_TRAINING_PARAMS
from movie_classifier.scripts.pipeline.training import training_pipeline

logger = logging.getLogger('Training API')


@app.route('/train', methods=['POST'])
def training():

    params = request.json

    if params is None:
        params = DEFAULT_TRAINING_PARAMS

    logger.info(' > It may take several minutes. Check logs of the api server to keep track of the training process')
    model = training_pipeline(params)

    return make_response(jsonify({'Status': 'OK'}))
