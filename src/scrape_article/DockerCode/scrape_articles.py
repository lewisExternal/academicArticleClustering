from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.chrome.options import Options
import os
import sys
import io
#from webdriver_manager.chrome import ChromeDriverManager
import numpy as np 
import pandas as pd 
import configparser
import re 
import PyPDF2
from  time import sleep
import glob
import shutil 
import logging
import json

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams

from multiprocessing.pool import Pool

# logging 
logging.basicConfig(filename='/logs/scrape_articles.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s')
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

def read_config(name):
    # get config
    config = configparser.ConfigParser()
    config.read('./config.ini')
    save_pdf_flag = str(config.get(name, 'save_pdf_flag')).lower()
    if save_pdf_flag == 'true':
        save_pdf_flag = True 
    else:
        save_pdf_flag = False 
    return save_pdf_flag

def get_lookup_link(index: int = 0) -> str:
    config = configparser.ConfigParser()
    config.read('./config.ini')
    lookup_links = config.get('main', 'url_lookup')
    lookup_links = lookup_links.split(',')
    lookup_link = lookup_links[index]
    lookup_link = 'https://' + lookup_link + '/' 
    return lookup_link

def get_driver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--window-size=1420,1080')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-dev-shm-usage')        
    driver = webdriver.Chrome(chrome_options=chrome_options)
    return driver

def aggregate_list(start_list):
    final_list = []
    for list_element in start_list:
        if len(list_element) > 0:
            for element in list_element:
                final_list.append(element)
    return final_list

def save_pdf(pdf_lookup,attempts=3):
    driver  = get_driver()
    results = []
    #pdf_lookup = 'https://www.sci-hub.st/'+ pdf_lookup[0]
    pdf_lookup = get_lookup_link() +  pdf_lookup
    for attempt in range(attempts):
        try:
            driver.get(pdf_lookup)
            save_pdf = driver.find_element_by_id('buttons')
            save_pdf = driver.find_element_by_tag_name('button')
            save_pdf.click()
            logging.info(pdf_lookup)
            return 'Success'
        except Exception as e:
            print(str(e))
            driver.refresh()
            sleep(2)
    return 'Error'

# def pdf_to_text(file_name):
#     pdf  = PyPDF2.PdfFileReader(file_name)
#     num_pages = pdf.getNumPages()
#     result = ''
#     for page in range (num_pages):
#         page_text = pdf.getPage(int(page)).extractText()
#         result = result + ' ' + page_text
#     result =  cleanse_pdf_text(result)
#     return result 

def cleanse_pdf_text(text):
    text_cleansed = text.replace('\n',',')
    return text_cleansed

# def pdf_extract_info(file_name):
#     # initialise values 
#     result = {}
#     content = ''
#     try: 
#         pdf = PyPDF2.PdfFileReader(file_name)
#         info = pdf.getDocumentInfo()
#         result = {**info}
#         pages = int(pdf.getNumPages())
#         for page in range(pages):
#             content = content + str(pdf.getPage(page).extractText().replace('\n',''))
#         result['content'] = content  
#     except Exception as e:
#         print(str(e))
#     return result 

def pdfparser(file_name):
    # initialise values 
    result = {}
    content = ''
    try: 
        pdf = PyPDF2.PdfFileReader(file_name)
        info = pdf.getDocumentInfo()
        result = {**info}
    except Exception as e:
        logging.info(str(e))       
    try: 
        fp = open(file_name, 'rb')
        rsrcmgr = PDFResourceManager()
        retstr = io.StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        for page in PDFPage.get_pages(fp):
            interpreter.process_page(page)
        result['content'] = retstr.getvalue()
    except Exception as e:
        logging.info(str(e))
    return result 

def process_dois(dois, save_pdf_flag, splits:int=10, processes:int=4):
    links_len = int(len(dois))
    logging.info('Links to process: '+ str(links_len))
    logging.info('Number of splits: '+ str(splits))
    # partition dataset 
    for sub_array in np.array_split(dois, splits):
        # sub array length 
        sub_array_len = int(len(sub_array))
        logging.info('Processing links : '+ str(sub_array_len))       
        try:
            # multiprocessing 
            with Pool(processes) as p:
                p.map(save_pdf, sub_array)   
            logging.info('pdfs downloaded.')
            # convert pdf to txt 
            convert_pdf_to_txt(sub_array_len, save_pdf_flag=save_pdf_flag)
            logging.info('Conversion to txt files complete.')
            # log processed dois 
            save_processed_dois(sub_array)
            logging.info('Dois have been logged.')
        except Exception as e:
            logging.info('ERROR aborting: '+ str(e))

def convert_pdf_to_txt(sub_array_len: int, exit_cond: int=3, save_pdf_flag=False):
    # wait 1 second 
    sleep(1)
    # initialise value 
    counter = 0 
    exit_count = 0
    # loop until all pdf files are processed or n consec iterations 
    while (counter <= sub_array_len) and (exit_count <= exit_cond):
        files = os.listdir("./")
        # move files
        for file_name in files:
            if file_name.lower()[-4:] == ".pdf":
                txt_file_name =  file_name.replace(".pdf",".txt")
                result = pdfparser(file_name)
                with open(txt_file_name, 'w') as file:
                    file.write(json.dumps(result))
                # save pdf or remove 
                if save_pdf_flag:
                    shutil.move("./"+ file_name, '/result/pdf_raw/'+file_name)
                else:
                    os.remove("./"+ file_name)
                shutil.move("./"+ txt_file_name, '/result/pdf_transforms/'+txt_file_name)
                logging.info('The following file has been processed: ' + file_name)
                counter += 1 
        exit_count += 1 
        # wait 1 second 
        sleep(1)

def remove_processed(dois, save_loc:str='/logs/processed_dois.csv'):
    # lookup processed dois 
    try:
        df = pd.read_csv(save_loc)
        processed_dois = df['doi']
        # remove processed
        result = []
        logging.info('Original number of links: ' + str(len(dois)))
        for doi in dois:
            if doi not in list(processed_dois):
                result.append(doi)
        logging.info('Links to process : ' + str(len(result)))
        return result
    except Exception as e:
        logging.info('lookup has failed error return full list: ' + str(e))
        return dois

def save_processed_dois(dois, save_loc:str='/logs/processed_dois.csv', bu_save_loc:str='/logs/bu_processed_dois.csv'):
    try:
        #breakpoint()
        df = pd.read_csv(save_loc)
        processed_dois = df['doi']
        processed_dois = list(processed_dois)
    except Exception as e:
        df = pd.DataFrame(columns=['doi'])
        processed_dois = []
        logging.info('lookup failed.')
    for doi in dois:
        processed_dois.append(doi)    
    processed_dois = np.unique(processed_dois)
    df = pd.DataFrame(processed_dois, columns=['doi'])
    df.to_csv(save_loc)
    if len(processed_dois)>0:
        df.to_csv(bu_save_loc)

def main():

    # read specific config 
    save_pdf_flag = read_config('main')
    
    # read in doi names 
    doi_df = pd.read_csv('/result/dois.csv')
    dois = doi_df['doi']

    # check processed pdfs 
    lookup_dois = remove_processed(dois)

    # try create sub dirs
    try:  
        os.makedirs('/result/pdf_raw', exist_ok=True)
    except Exception as e:
        logging.info('ERROR: '+ str(e))
    try:  
        os.makedirs('/result/pdf_transforms', exist_ok=True)
    except Exception as e:
        logging.info('ERROR: '+ str(e))

    # get pdfs 
    process_dois(lookup_dois, save_pdf_flag)    

logging.info('script finished')

if __name__ == "__main__":
    main()
