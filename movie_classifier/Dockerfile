FROM python:3.7.5-buster

WORKDIR /app

COPY ./venv/lib/python3.8/site-packages/movie_classifier /app/movie_classifier

EXPOSE 5000

ENV FLASK_APP=/app/movie_classifier/api_server.py
ENV PYTHONPATH=${PYTHONPATH}:/app/movie_classifier

RUN pip3 install --upgrade pip
RUN pip3 install -r movie_classifier/requirements.txt

ENTRYPOINT ["python3"]
CMD ["-m" , "flask", "run", "--host=0.0.0.0"]
