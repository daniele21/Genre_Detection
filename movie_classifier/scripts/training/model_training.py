from typing import Dict, Text

from movie_classifier.core.preprocessing.tokenizers import MyTokenizer
from movie_classifier.scripts.network.network_init import init_network


def train_model(dataset: Dict[Text, Dict],
                tokenizer: MyTokenizer,
                params: Dict,
                callbacks=None):

    network_params = params['network']
    training_params = params['training']

    network_params['n_word_tokens'] = tokenizer.n_words + 2
    network_params['n_classes'] = tokenizer.n_labels

    model = init_network(network_params, tokenizer, compile=True)

    x_train, y_train = dataset['train']['x'], dataset['train']['y']
    x_test, y_test = dataset['test']['x'], dataset['test']['y']

    batch_size = training_params['batch_size']
    epochs = training_params['epochs']

    model.fit(x=x_train, y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=callbacks,
              verbose=1,
              )

    return model


