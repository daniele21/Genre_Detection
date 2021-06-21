# Movie Classifier Project

![Licence](https://img.shields.io/badge/Licence-MIT-orange)

Libraries: 

![Tensorflow](https://img.shields.io/badge/Tensorflow-2.5-brightgreen)
![Flask](https://img.shields.io/badge/Flask-2.0.1-brightgreen)
![NLTK](https://img.shields.io/badge/NLTK-3.6.2-brightgreen)
![Pandas](https://img.shields.io/badge/Pandas-1.2.4-brightgreen)

Dependences:

[![Python](https://img.shields.io/badge/Python-3.8-yellow)](https://github.com/daniele21/movie_classifier/blob/master/dependences.md)
[![Docker](https://img.shields.io/badge/Docker-20.10.5-blue)](https://github.com/daniele21/movie_classifier/blob/master/dependences.md)

## Contents
- [Description](#description)
- [Dataset](#dataset)
- [Task](#task)
- [Solution](#solution)
- [Try it!](#try-it)

------------------------

# Description
The project goal was to create a **Deep Learning** model able to detect the **movie genre**, given a *title* and a *description*

# Dataset
The dataset is located inside *resources* folder. You can find **train.csv** and **test.csv**.

Data features: **movie_id, title, year, genres, synopsis**

Have a look: [dataset](https://raw.githubusercontent.com/daniele21/movie_classifier/master/resources/test.csv)

# Task
Text Classification

# Solution
**Deep Learning** model for text classification, based on **LSTM Network**. The model uses a **Word Embedding** initialized with *random values*, but it is possible to test it with an pretrained embedding

------------------------

# Try it!

### 0. Check Dependences

- python >= 3.8
- docker >= 20

If needed, install them from [how to install dependeces](https://github.com/daniele21/movie_classifier/blob/master/dependences.md)

### 1. Create a virtual environment

    python3.8 -m venv ./venv
    source venv/bin/activate

### 2. Download the project package and Install it
    
download the movie classifier library: [movie_classifier-0.1.tar.gz](https://drive.google.com/file/d/1cHsATW9hWoMWTaryPYmPrm_pKfUPVvML/view?usp=sharing)
    
    pip install path/to/movie_classifier-0.1.tar.gz

### 3. Run the Docker API Server
    
    movie_classifier --api-server

### 4. Enjoy your test
   
*command*: **movie_classifier**

*options*:
- **--title** <MOVIE_TITLE>
- **--description** <MOVIE_DESCRIPTION>
  
example: 

    movie_classifier --title 'othello' --description 'some othello description'

### 5. If you need, train by yourself

    movie_classifier --train  
    
   (wait until it finishes - it could take some minutes, it depends on your hardware, check the docker logs for training details)

You can check the training logs through the docker logs accessible with:
    
    docker ps | grep 'movie_classifier' | awk '{ print $1 }' > .CONTAINER_ID
    docker logs -f $(cat .CONTAINER_ID) 

Automatically during the test, it will be loaded the **most recent trained model**

### 6. Custom Training: you choose the hyperparameters

Change the following hyperparameters as you want and run the curl command

    curl --location --request POST 'http://localhost:5000/train' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "data": {
            "split_size": 0.7,
            "seed": 2021
        },
        "network": {
            "lstm_units": 128,
            "dropout_rate": 0.3,
            "lr": 0.001,
        },
        "training": {
            "batch_size": 64,
            "epochs": 15
        }
    }'

------------------------



#### Author
Daniele Moltisanti

[![Linkedin](https://img.shields.io/badge/Linkedin-Daniele%20Moltisanti-blue)](https://www.linkedin.com/in/daniele-moltisanti/)

