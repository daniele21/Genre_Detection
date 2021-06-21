import unittest

from movie_classifier.constants.params import DEFAULT_TRAINING_PARAMS
from movie_classifier.core.utils.logger import setup_logger_config
from movie_classifier.scripts.pipeline.training import training_pipeline

setup_logger_config()


class DatasetPipelineTest(unittest.TestCase):
    def test_training(self):
        params = DEFAULT_TRAINING_PARAMS
        # params['training']['epochs'] = 1

        model = training_pipeline(params)

        self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
