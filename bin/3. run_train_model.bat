@echo off
cd ../src/train_model/
docker-compose build --no-cache
docker-compose up -d
docker attach train_model
pause