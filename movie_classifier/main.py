import json
from argparse import ArgumentParser
import logging
import requests

from movie_classifier.core.utils.prints import dict_print
from movie_classifier.scripts.utils.checker import check_main_args
from movie_classifier.core.utils.logger import setup_logger_config
from movie_classifier.docker_api_server import run_docker_api_server


setup_logger_config()

TRAINING_ENDPOINT = 'http://localhost:5000/train'
PREDICT_ENDPOINT = 'http://localhost:5000/predict'

logger = logging.getLogger()


def movie_classifier_process(args):

    check_main_args(args)

    print(args)

    if args.api_server:
        run_docker_api_server()

    elif args.train:
        params = None       # It takes the default parameters
        logger.info(f' > Training')
        logger.debug(f' Training params: {params}')

        response = requests.post(TRAINING_ENDPOINT, json=params)

        print(response.text)

    else:
        title, description = args.title, args.description
        data = {'title': title,
                'description': description}

        logger.info(' > Predict')
        response = requests.post(PREDICT_ENDPOINT, json=data)
        json_resp = json.loads(response.text)

        outcome = {'title': json_resp['title'],
                   'description': json_resp['description'],
                   'genre': json_resp['genre']}

        print('')
        print('Genre Detection output:\n')
        dict_print(outcome)
        print('')


def main():
    parser = ArgumentParser()

    parser.add_argument('--api-server', dest='api_server', action='store_true',
                        help='Run the api server on docker container')

    parser.add_argument('-t', '--title', type=str, help='Provide a movie title')
    parser.add_argument('-d', '--description', type=str, help='Provide a movie description')

    parser.add_argument('--train', dest='train',
                        action='store_true', help="set training mode")

    parsed_args = parser.parse_args()

    movie_classifier_process(parsed_args)


if __name__ == '__main__':
    main()
