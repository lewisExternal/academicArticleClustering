@echo off
cd ../src/scrape_dois/
docker-compose build --no-cache
docker-compose up -d
docker attach scrape_doi
pause