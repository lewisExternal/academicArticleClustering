#!/bin/bash
# to run as a background process use the below 
# nohup COMMAND &>/dev/null &
cd ../src/scrape_dois/
docker-compose build --no-cache
docker-compose up -d
docker attach scrape_doi
cd ../scrape_article/
docker-compose build --no-cache
docker-compose up -d 
docker attach scrape_article
cd ../train_model/
docker-compose build --no-cache
docker-compose up -d
docker attach train_model