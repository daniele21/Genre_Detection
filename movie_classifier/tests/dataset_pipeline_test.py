import unittest

from movie_classifier.constants.paths import TRAIN_PATH
from movie_classifier.core.utils.logger import setup_logger_config
from movie_classifier.scripts.pipeline.dataset_generation import generate_training_dataset

setup_logger_config()


class DatasetPipelineTest(unittest.TestCase):
    def test_dataset_training_generation(self):
        params = {'train': True,
                  'split_size': 0.8,
                  'shuffle': True,
                  'seed': 2021,
                  'data_path': TRAIN_PATH,
                  }

        dataset, tokenizer = generate_training_dataset(params)

        self.assertIsNotNone(dataset)
        self.assertIsNotNone(tokenizer)


if __name__ == '__main__':
    unittest.main()
