FROM devourey/scikit-sandbox:latest

COPY ./ml_app/requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

RUN mkdir /app
# COPY files in the host machine to the container
COPY ./ml_app /app

# Change into the backend directory
WORKDIR /app
