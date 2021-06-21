from subprocess import call
import logging
import movie_classifier

logger = logging.getLogger('Docker Api Server')

DOCKERFILE_PATH = f"{movie_classifier.__path__[0]}/Dockerfile"


def run_docker_api_server():

    logger.info(f' > Creating docker image for Movie Classifier')
    call(["docker", "build", "-f", f"{DOCKERFILE_PATH}", "-t", "movie_classifier_image", ".", "--no-cache"])
    logger.info(f' Image Created\n\n')

    logger.info(f' > Running the docker container')
    call(["docker", "run", "-d", "-p", "5000:5000", "-t", "-i", "movie_classifier_image"])

