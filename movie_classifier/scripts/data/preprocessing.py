import pandas as pd

from movie_classifier.core.preprocessing.text_preprocessing import remove_punctuations, remove_stopwords, \
    stem_sentence, lemmatize_sentence
from movie_classifier.core.utils.time_decorator import timing


@timing
def sentence_preprocessing(input_data: pd.DataFrame,
                           stemming=True,
                           lemmatization=False,
                           lowercase=True,
                           stopwords=True,
                           ):
    data = input_data.copy(deep=True)

    # REMOVING PUNCTUATIONS
    data['synopsis'] = data['synopsis'].apply(remove_punctuations)

    # LOWERING
    if lowercase:
        data['synopsis'] = data['synopsis'].apply(lambda x: str(x).lower())

    # REMOVING STOPWORDS
    if stopwords:
        data['synopsis'] = data['synopsis'].apply(remove_stopwords)

    # STEMMING
    if stemming:
        data['synopsis'] = data['synopsis'].apply(stem_sentence)

    # LEMMATIZATION
    if lemmatization:
        data['synopsis'] = data['synopsis'].apply(lemmatize_sentence)

    return data


def sentence_preprocessing_from_text(input_data: str,
                                     stemming=True,
                                     lemmatization=False,
                                     lowercase=True,
                                     stopwords=True,
                                     ):

    # REMOVING PUNCTUATIONS
    data = remove_punctuations(input_data)

    # LOWERING
    if lowercase:
        data = data.lower()

    # REMOVING STOPWORDS
    if stopwords:
        data = remove_stopwords(data)

    # STEMMING
    if stemming:
        data = stem_sentence(data)

    # LEMMATIZATION
    if lemmatization:
        data = lemmatize_sentence(data)

    return data
