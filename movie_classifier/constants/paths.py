from movie_classifier.core.utils.time_handler import readable_timestamp

TRAIN_PATH = 'movie_classifier/resources/train.csv'
TEST_PATH = 'movie_classifier/resources/test.csv'

MODELS_DIR = 'movie_classifier/resources/models/'
MODEL_NAME = f'LSTM_model_{readable_timestamp()}'
MODEL_DIR = f'{MODELS_DIR}{MODEL_NAME}/'
MODEL_PATH = f'{MODEL_DIR}{MODEL_NAME}.hdf5'

PARAMS_FILENAMES = {'data': 'data_params.json',
                    'network': 'network_params.json',
                    'training': 'training_params.json'}

GLOVE_PATH = 'resources/embeddings/glove.6B/glove.6B.300d.txt'

DOCKERFILE_PATH = 'movie_classifier/Dockerfile'
