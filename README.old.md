# Movie Classifier Project

![Licence](https://img.shields.io/badge/Licence-MIT%20Licence-green)

Libraries: 

![Tensorflow](https://img.shields.io/badge/Tensorflow-2.5-orange)
![Flask](https://img.shields.io/badge/Flask-2.0.1-red)
![NLTK](https://img.shields.io/badge/NLTK-3.6.2-green)
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

# Task
Text Classification

# Solution
**Deep Learning** model for text classification, based on **LSTM Network**. The model uses a **Word Embedding** initialized with *random values*, but it is possible to test it with an pretrained embedding

------------------------

# Try it!


### 1. Clone this repository
   
    git clone https://github.com/daniele21/movie_classifier.git
    cd movie_classifier

### 2. Run the initialization command

    ./init.sh

### 3. Run the Docker API Server
    
    ./run_api_server.sh
       

### 4. Active the virtual environment
    
    source venv/bin/activate
    
### 5. Export the path of the project

    export PYTHONPATH=$PYTHONPATH:$(pwd)
     
### 6. Train a model

    movie_classifier --train  
    
   (wait until it finishes - it could take some minutes, it depends on your hardware, check the docker logs for training details)

### 6. Enjoy your test
   
*command*: **movie_classifier**

*options*:
- **-t** <MOVIE_TITLE>
- **-d** <MOVIE_DESCRIPTION>
  
example: 

    movie_classifier -t 'othello' -d 'some othello description'

------------------------

#### Author
Daniele Moltisanti

[![Linkedin](https://img.shields.io/badge/Linkedin-Daniele%20Moltisanti-blue)](https://www.linkedin.com/in/daniele-moltisanti/)

