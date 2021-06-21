#!/bin/bash

# Running Docker API Server
echo ''
echo '    > Creating Docker Image'
echo ''
docker build -t movie_classifier_docker .
echo ''
echo '    > Docker image created under the following name: movie_classifier_docker'

echo ''
echo '    > Starting the Movie Classifier container'
docker run -p 5000:5000 -t -i movie_classifier_docker

