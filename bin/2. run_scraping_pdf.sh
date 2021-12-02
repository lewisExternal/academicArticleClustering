#!/bin/bash
cd ../src/scrape_article/
docker-compose build --no-cache
docker-compose up -d 
docker attach scrape_article