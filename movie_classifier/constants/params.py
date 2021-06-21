from keras.callbacks import EarlyStopping, ModelCheckpoint

from movie_classifier.constants.paths import TRAIN_PATH, MODEL_PATH

MAX_WORD_SENTENCE = 220

ADAM_OPTIMIZER = 'adam'
BCE_LOSS = 'bce'

DEFAULT_TRAINING_PARAMS = {'data': {'train': True,
                                    'split_size': 0.7,
                                    'shuffle': True,
                                    'seed': 2021,
                                    'data_path': TRAIN_PATH,
                                    },

                           'network': {'word_emb_size': 300,
                                       # 'weights': None,
                                       'trainable': True,
                                       'lstm_units': 128,
                                       'dropout_rate': 0.3,
                                       'optimizer': ADAM_OPTIMIZER,
                                       'lr': 0.001,
                                       'loss': BCE_LOSS},

                           'training': {'batch_size': 64,
                                        'epochs': 15}

                           }

# {
#     "data": {
#         "split_size": 0.7,
#         "shuffle": true,
#         "seed": 2021
#     },
#     "network": {
#         "word_emb_size": 300,
#         "trainable": true,
#         "lstm_units": 128,
#         "dropout_rate": 0.3,
#         "optimizer": "adam",
#         "lr": 0.001,
#         "loss": "bce"
#     },
#     "training": {
#         "batch_size": 64,
#         "epochs": 15
#     }
# }

DEFAULT_CALLBACKS = [EarlyStopping(monitor='val_loss', patience=1, verbose=1),
                     ModelCheckpoint(filepath=MODEL_PATH, save_weights_only=False,
                                     monitor='val_acc', save_best_only=True,
                                     mode='min', verbose=0),
                     ]
