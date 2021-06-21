import unittest

from movie_classifier.constants.paths import MODELS_DIR
from movie_classifier.scripts.model.inference import genre_prediction
from movie_classifier.scripts.pipeline.dataset_generation import generate_test_sample
from movie_classifier.scripts.utils.loadings import load_latest_model, load_tokenizer


class InferenceTests(unittest.TestCase):
    def test_model_loading(self):

        model = load_latest_model(MODELS_DIR)

        self.assertIsNotNone(model)

    def test_model_inference(self):
        model, model_dir = load_latest_model(MODELS_DIR)
        tokenizer = load_tokenizer(model_dir)

        description = 'Once upon a time'
        descr_sample = generate_test_sample(description, tokenizer)
        genre = genre_prediction(model, descr_sample, tokenizer)
        print(genre)

        self.assertIsNotNone(genre)


if __name__ == '__main__':
    unittest.main()
