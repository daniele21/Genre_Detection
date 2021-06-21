from typing import Dict, Text

from movie_classifier.constants.params import DEFAULT_CALLBACKS, DEFAULT_TRAINING_PARAMS
from movie_classifier.constants.paths import MODEL_DIR
from movie_classifier.scripts.pipeline.dataset_generation import generate_training_dataset
from movie_classifier.scripts.training.model_training import train_model
from movie_classifier.scripts.visualization.plots import plot_loss


def training_pipeline(params: Dict[Text, Dict],
                      save_dir=MODEL_DIR,
                      callbacks=DEFAULT_CALLBACKS):

    params['data']['train'] = True

    params = check_params(params)
    data_params = params['data']

    # Getting training dataset
    dataset, tokenizer = generate_training_dataset(data_params, save_dir=save_dir)

    # Training model
    model = train_model(dataset, tokenizer, params, callbacks=callbacks)

    plot_loss(model.history.history['loss'], model.history.history['val_loss'])

    return model


def check_params(given_params):

    data_params = {'split_size': given_params['split_size'] if 'split_size' in given_params else DEFAULT_TRAINING_PARAMS['data']['split_size'],
                   'shuffle': DEFAULT_TRAINING_PARAMS['data']['shuffle'],
                   'seed': given_params['seed'] if 'seed' in given_params else DEFAULT_TRAINING_PARAMS['data']['seed'],
                   'data_path': DEFAULT_TRAINING_PARAMS['data']['data_path']}

    network_params = {'word_emb_size': given_params['word_emb_size'] if 'word_emb_size' in given_params else DEFAULT_TRAINING_PARAMS['network']['word_emb_size'],
                      'trainable': DEFAULT_TRAINING_PARAMS['network']['trainable'],
                      'lstm_units': given_params['lstm_units'] if 'lstm_units' in given_params else DEFAULT_TRAINING_PARAMS['network']['lstm_units'],
                      'dropout_rate': given_params['dropout_rate'] if 'dropout_rate' in given_params else DEFAULT_TRAINING_PARAMS['network']['dropout_rate'],
                      'optimizer': DEFAULT_TRAINING_PARAMS['network']['optimizer'],
                      'loss': DEFAULT_TRAINING_PARAMS['network']['loss'],
                      'lr': given_params['lr'] if 'lr' in given_params else DEFAULT_TRAINING_PARAMS['network']['lr']
                      }
    training_params = {'batch_size': given_params['batch_size'] if 'batch_size' in given_params else DEFAULT_TRAINING_PARAMS['training']['batch_size'],
                       'epochs': given_params['epochs'] if 'epochs' in given_params else DEFAULT_TRAINING_PARAMS['training']['epochs']
                       }

    params = {'data': data_params,
              'network': network_params,
              'training': training_params}

    return params

