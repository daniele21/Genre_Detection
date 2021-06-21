import os

import keras

from movie_classifier.constants.paths import PARAMS_FILENAMES, MODELS_DIR
from movie_classifier.core.file_manager.loadings import load_json, pickle_load


def load_model(model_dir):
    model_name = model_dir.split('/')[-2]
    model_path = f'{model_dir}{model_name}.hdf5'

    return keras.models.load_model(model_path)


def load_latest_model(models_dir=MODELS_DIR):
    models_folder = os.listdir(models_dir)
    models_folder_path = [f'{models_dir}{x}/' for x in models_folder if str(x).startswith('LSTM_model')]
    latest_model = max(models_folder_path, key=os.path.getctime)
    model_name = latest_model.split('/')[-2]

    model_path = f'{latest_model}{model_name}.hdf5'
    # logger.info(f' > Model_path: {model_path}')
    print(f' > Model_path: {model_path}')

    return keras.models.load_model(model_path), latest_model


def load_params(type, model_dir):
    """

    :param type:            ['data' | 'network' | 'training']
    :param model_dir:
    :return:
    """

    filepath = f'{model_dir}{PARAMS_FILENAMES[type]}'
    params = load_json(filepath)

    return params


def load_tokenizer(model_dir):
    filepath = f'{model_dir}tokenizer'
    return pickle_load(filepath)
