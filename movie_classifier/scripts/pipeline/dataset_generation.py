from typing import Text, Dict

from movie_classifier.core.file_manager.os_utils import ensure_folder
from movie_classifier.core.file_manager.savings import pickle_save
from movie_classifier.core.preprocessing.text_preprocessing import init_nltk
from movie_classifier.core.preprocessing.tokenizers import CustomTokenizer, MyTokenizer
from movie_classifier.core.utils.time_decorator import timing
from movie_classifier.scripts.data.data_loading import load_data
from movie_classifier.scripts.data.dataset import split_data, create_dataset, create_inference_dataset, create_single_sentence_dataset
from movie_classifier.scripts.data.preprocessing import sentence_preprocessing, sentence_preprocessing_from_text

import numpy as np
import logging

logger = logging.getLogger('Dataset Generation')


@timing
def generate_training_dataset(params: Dict,
                              save_dir: Text = None):
    """

    :param save_dir:
    :param params:  dict {
                          'train':      True,
                          'split_size:  double,
                          'shuffle':    bool,
                          'seed':       int,
                          'data_path:'  str,
                          }
    :return:
        x_train, x_test, y_train, y_test
    """

    # Loading data
    data = load_data(params['data_path'])

    init_nltk()

    # Sentence Preprocessing for synopsis
    prep_data = sentence_preprocessing(data,
                                       stemming=True,
                                       lemmatization=False,
                                       lowercase=True,
                                       stopwords=True,
                                       # preload=params.get('preload'),
                                       )

    # Loading tokenizer function
    tokenizer = CustomTokenizer()

    # Tokenizer fit
    tokenizer.fit(prep_data['synopsis'], prep_data['genres'])

    # Dataset generation
    x_dataset, y_dataset = create_dataset(prep_data['synopsis'], prep_data['genres'], tokenizer)

    # Splitting data between train and test sets
    x_train, x_test, y_train, y_test = split_data(x_dataset, y_dataset, params)

    logger.info('Dataset Shapes:')
    logger.info(f'\tX_Train: {x_train.shape}\t-\t X_Test : {x_test.shape}')
    logger.info(f'\ty_Train: {y_train.shape}\t-\t y_Test : {y_test.shape}')

    dataset = {'train': {'x': x_train,
                         'y': y_train},
               'test': {'x': x_test,
                        'y': y_test}}

    # Saving Tokenizer
    if save_dir is not None:
        filepath = f'{save_dir}tokenizer'
        ensure_folder(save_dir)
        pickle_save(tokenizer, filepath)

    return dataset, tokenizer


def generate_test_dataset(data_path, tokenizer):
    data = load_data(data_path)
    init_nltk()
    prep_data = sentence_preprocessing(data,
                                       stemming=True,
                                       lemmatization=False,
                                       lowercase=True,
                                       stopwords=True)

    dataset = create_inference_dataset(prep_data['synopsis'], tokenizer)

    return data, np.array(dataset)


def generate_test_sample(text: str,
                         tokenizer: MyTokenizer):
    init_nltk()
    prep_data = sentence_preprocessing_from_text(text,
                                                 stemming=True,
                                                 lemmatization=False,
                                                 lowercase=True,
                                                 stopwords=True)

    dataset = create_single_sentence_dataset(prep_data, tokenizer)

    return np.array(dataset)