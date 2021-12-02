# set base image (host OS)
FROM python:3.8

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# download corpus 
RUN python -m textblob.download_corpora

# download stop words 
RUN python -m nltk.downloader stopwords

# copy the content of the local src directory to the working directory
COPY /DockerCode/ .

# command to run on container start
CMD [ "python", "./train_model.py" ]