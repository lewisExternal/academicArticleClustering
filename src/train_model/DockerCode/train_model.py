import os
import sys
import io
#from webdriver_manager.chrome import ChromeDriverManager
import numpy as np 
import pandas as pd 
import configparser
import re 
#import PyPDF2
from  time import sleep
import glob
import shutil 
import logging
import json
from langdetect import detect
from nltk.corpus import stopwords
from textblob import TextBlob, Word, Blobber
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn import metrics
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from itertools import cycle
from wordcloud import WordCloud, ImageColorGenerator


# logging 
logging.basicConfig(filename='/logs/train_model.log', level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s')
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# create sub dirs
try:  
    os.makedirs('/result/model', exist_ok=True)
except Exception as e:
    logging.info('ERROR: '+ str(e))

def read_config(name:str='main'):
    """  reads config and retruns parameters """
    config = configparser.ConfigParser()
    config.read('./config.ini')
    k = int(config.get(name, 'k'))
    k_range = int(config.get(name, 'k_range'))
    return k, k_range 

def read_json_files()->pd.DataFrame:
    """ reads from txt files and loads into a dataframe """
    result = None
    lookup_path = "/result/pdf_transforms/"
    files = os.listdir(lookup_path)
    for file_name in files:
        if file_name.lower()[-4:] == ".txt":
            try:
                with open(lookup_path+file_name) as json_file:
                    data = json.load(json_file)
                    data['file_name'] = file_name
                dataframe = pd.DataFrame(data, index=[0]) 
                if result is not None:
                    result = result.append(dataframe)
                else:
                    result = dataframe
            except Exception as e:
                logging.info('ERROR: '+ str(e))

    return result

def validate_config(dataframe: pd.DataFrame, k: int=0, k_range:int=0)->None:
    """ check config values meet criteria """
    samples = int(dataframe.shape[0])
    logging.info('The number of samples are: '+ str(samples))
    if not samples > 0:
        sys.exit(1) 
    # check k values 
    if samples <= min(k_range,k):
        sys.exit(1) 

def data_pre_processing(dataframe: pd.DataFrame)->None: 
    """ pre cleanse the article content """
    # remove duplicates 
    dataframe.drop_duplicates(['content'], inplace=True)
    # remove na values 
    #dataframe.dropna(inplace=True)

def article_cleansing(article: str='')->str:
    """ cleanse article text  """
    article = article['content']
    # only process english articles 
    if not is_english(article):
        return ''
    else: 
        # lower 
        cleansed = article.lower()

        # remove line break 
        cleansed = cleansed.replace('\n',' ')

        # specific encoding issue 
        cleansed = cleansed.replace('''\\u00ac''',' ')
        cleansed = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', cleansed)
        
        # remove punctuation
        cleansed = ''.join([z for z in cleansed if z not in string.punctuation])

        # remove digits 
        cleansed = ''.join([z for z in cleansed if z if not z.isdigit()])
        
        # remove stop words
        stop = stopwords.words('english')
        cleansed = ' '.join([z for z in cleansed.split(' ') if z if not z in stop])

        # lemmatize 
        cleansed = " ".join([Word(word).lemmatize() for word in cleansed.split()])

    return cleansed

def is_english(article: str)->bool:
    """ rerun true if string is in english else false """ 
    try:
        language  = detect(article)
    except Exception as e:
        language = ''
    if language == 'en':
        return True 
    else:
        return False  

def vectorize(text: str, max_features:int=10000):
    """ vectorize text """
    vectorizer = TfidfVectorizer(max_features=max_features)
    result = vectorizer.fit_transform(text)
    return result, result.shape
    
def principle_component_analysis(text, n_components: float=0.95): 
    """ reduce features to keep n_components*100 % of the variance """
    pca=PCA(n_components=n_components, random_state=42)
    logging.info('Fit PCA')
    logging.info(text)
    result=pca.fit_transform(text.toarray())
    return result, result.shape

def k_means_graph(text, reduced_text, k_range: int=10)->None:
    """ plot elbow and save figure """
    # run kmeans with many different k
    distortions = []
    K = range(2, k_range)
    for k in K:
        k_means = KMeans(n_clusters=k, random_state=42).fit(reduced_text)
        k_means.fit(reduced_text)
        distortions.append(sum(np.min(cdist(reduced_text, k_means.cluster_centers_, 'euclidean'), axis=1)) / text.shape[0])
        #print('Found distortion for {} clusters'.format(k))
    X_line = [K[0], K[-1]]
    Y_line = [distortions[0], distortions[-1]]
    # Plot the elbow to find k 
    plt.plot(K, distortions, 'b-')
    plt.plot(X_line, Y_line, 'r')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig("/result/model/k_means_elbow.png")
    plt.clf()

def run_k_means(reduced_text: str, k: int, dataframe: pd.DataFrame)->None:
    """ create dataframe column for k means cluster """
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(reduced_text)
    dataframe['y_pred'] = y_pred
    return y_pred

def tsne(text:str):
    """ fit tsne for 2d graph """
    tsne = TSNE(verbose=1, perplexity=100, random_state=42)
    result = tsne.fit_transform(text.toarray())
    return result 

def lda(k: int, dataframe, NUM_TOPICS_PER_CLUSTER: int=20):
    """ fit lda """
    vectorizers = []
    for ii in range(0, k):
        # Creating a vectorizer
        vectorizers.append(CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}'))
    vectorized_data = []
    for current_cluster, cvec in enumerate(vectorizers):
        try:
            vectorized_data.append(cvec.fit_transform(dataframe.loc[dataframe['y_pred'] == current_cluster, 'cleansed']))
        except Exception as e:
            print("Not enough instances in cluster: " + str(current_cluster))
            vectorized_data.append(None)
    lda_models = []
    for ii in range(0, k):
        # Latent Dirichlet Allocation Model
        lda = LatentDirichletAllocation(n_components=NUM_TOPICS_PER_CLUSTER, max_iter=10, learning_method='online',verbose=False, random_state=42)
        lda_models.append(lda)
    clusters_lda_data = []
    for current_cluster, lda in enumerate(lda_models):
        # print("Current Cluster: " + str(current_cluster))
        if vectorized_data[current_cluster] != None:
            clusters_lda_data.append((lda.fit_transform(vectorized_data[current_cluster])))
    # Functions for printing keywords for each topic
    return vectorized_data, lda_models, clusters_lda_data, vectorizers

def selected_topics(model, vectorizer, top_n=3):
    """ find cluster topics """
    current_words = []
    keywords = []
    for idx, topic in enumerate(model.components_):
        words = [(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.append(word[0])            
        keywords.sort(key = lambda x: x[1])  
        keywords.reverse()
        return_values = []
    for ii in keywords:
        return_values.append(ii[0])
    return return_values

def get_keywords(lda_models, vectorized_data, vectorizers):
    """ return key words within a given cluster """
    all_keywords = []
    logging.info('The number of lda models is: ' )
    for current_vectorizer, lda in enumerate(lda_models):
        # print("Current Cluster: " + str(current_vectorizer))
        if vectorized_data[current_vectorizer] != None:
            all_keywords.append(selected_topics(lda, vectorizers[current_vectorizer]))
    return all_keywords

def display_cloud(cluster_num, cmap, all_keywords):
    """ save word cloud """
    wc = WordCloud(background_color="black", max_words=2000, max_font_size=80, colormap=cmap)
    logging.info(all_keywords)
    wordcloud = wc.generate(' '.join([word for word in all_keywords[cluster_num]]))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('/result/model/wc_cluster_' + str(cluster_num), bbox_inches='tight')
    plt.clf()

def create_word_cloud(all_keywords, k: int):
    """ cycle through cluster word clouds """
    cmaps = cycle(['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])
    for i in range(k):
        logging.info("Processing cluster: " +str(i))
        cmap = next(cmaps)
        display_cloud(i, cmap, all_keywords)

def plot_data(text_embedded, y_pred):
    """ create the t-sne plot """
    # sns settings
    sns.set(rc={'figure.figsize':(15, 15)})
    # colors
    #palette = sns.hls_palette(20, l=.4, s=.9)
    palette = sns.color_palette('coolwarm', as_cmap = True)
    # plot
    sns.scatterplot(text_embedded[:,0], text_embedded[:,1], hue=y_pred, legend='full', palette=palette)
    plt.title('t-SNE with Kmeans Labels')
    plt.savefig("/result/model/improved_cluster_tsne.png")
    plt.show()
    plt.clf()

def main():

    # read from saved json files 
    dataframe = read_json_files()

    # read from config 
    k, k_range = read_config('main')

    # validate config 
    validate_config(dataframe, k, k_range)

    # data_pre_processing
    data_pre_processing(dataframe)
    
    # cleanse articles 
    dataframe['cleansed'] = dataframe.apply(article_cleansing, axis=1)

    # remove empty article text 
    dataframe = dataframe[dataframe['cleansed']!='']

    # vectorization
    text = dataframe['cleansed'].values
    text, shape = vectorize(text, 2 ** 12)
    logging.info('Shape is: '+ str(shape))
    
    # tokenize
    reduced_text, shape_reduced = principle_component_analysis(text)
    logging.info('Shape reduced is: '+ str(shape_reduced))

    # k means elbow graph 
    k_means_graph(text, reduced_text, k_range)
    
    # run k means 
    y_pred = run_k_means(reduced_text, k, dataframe)

    # t-SNE dimensionality reduction  
    text_embedded = tsne(text)

    # plot data 
    plot_data(text_embedded, y_pred)

    # LDA (Latent Dirichlet Allocation)
    vectorized_data, lda_models, clusters_lda_data, vectorizers = lda(k, dataframe)

    # get kewords 
    all_keywords = get_keywords(lda_models, vectorized_data, vectorizers)
    
    # create topic word clouds 
    create_word_cloud(all_keywords, k)
       
    logging.info(dataframe)
    logging.info('script finished')

if __name__ == "__main__":
    main()
