
# Academic Article Clustering and Mining 

This project is a proof of concept for mining and clustering academic articles using K Means. The script both looks up articles and converts the content to text by looking up a CSV of DOI numbers ( a standardized unique number given to many articles, papers, & books, by some publishers, to identify a particular publication ). The script will also mine these DOI numbers from jamanetwork.com. 

## Data

The DOI numbers numbers in the POC were searched from jamanetwork.com.

## How to run
Required scripts can be found within the bin/ subdirectory.
To run all scripts sequentally, the following can be run:

 - bin/0.run_scraping_all.bat (Windows)
 - bin/0.run_scraping_all.sh (Unix)

Otherwise, scripts can be run in isolation:

The below will generate a CSV file of DOI numbers from jamanetwork.com (i.e. result/dois.csv ) current configured to search for articles relating to "cancer". Though can be configured within the config.ini file. 

 - bin/1.run_scraping_doi.bat (Windows) 
 - bin/1.run_scraping_doi.sh (Unix)

A CSV file with DOIs must be provided for the result/dois.csv. The following will lookup PDFs using multi processing and converts them to dictionary objects which are saved to .txt files within direcotry: result\pdf_transforms where possible. 
- bin/2. run_scraping_pdf.bat
- bin/2. run_scraping_pdf.sh 

There must be processed articles within the result\pdf_transforms direcotry for the following to run. The below will read in the processed articles, cleanse the text, run TfidfVectorizer and then PCA. Create a k means elbow graph to determine suitability of our chosen k value (within config.ini). Runs k means clustering and uses t-SNE dimensionality reduction to view the article clusters in two dimensions. Then uses LDA (Latent Dirichlet Allocation) to generate word clouds for the saved clusters. 
 
- bin/3. run_train_model.bat
- bin/3. run_train_model.sh

This approach was derived in a large part from the following source [COVID19 literature clustering](https://maksimekin.github.io/COVID19-Literature-Clustering/COVID19_literature_clustering.html).

## Model Training

The model training will run K means clustering and produce the following.

### t-SNE plot 

### Elbow plot showing the optimal k 

### Word clouds 


## Requirements
 1.  [Docker](https://www.docker.com/products/personal)  
## References 
 1. https://nander.cc/using-selenium-within-a-docker-container
 2. https://github.com/SeleniumHQ/docker-selenium/blob/trunk/NodeChrome/Dockerfile
 3. https://newbedev.com/how-to-run-selenium-with-chrome-in-docker
 4. https://www.toptal.com/python/beginners-guide-to-concurrency-and-parallelism-in-python
 6. https://maksimekin.github.io/COVID19-Literature-Clustering/COVID19_literature_clustering.html
