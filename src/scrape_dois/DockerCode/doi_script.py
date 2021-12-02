from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.chrome.options import Options
import os
import sys
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

# logging 
logging.basicConfig(filename='/logs/doi_script.log', level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s')
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

def read_config(name: str):
    """ read from config file """
    # get config
    config = configparser.ConfigParser()
    config.read('./config.ini')
    config_dict = {}
    url = str(config.get(name, 'url'))
    search_limit = int(config.get(name, 'search_limit'))
    search_term = str(config.get(name, 'search_term'))
    return url, search_limit, search_term

def get_lookup_link(index: int = 0) -> str:
    """ lookup link to download article """
    config = configparser.ConfigParser()
    config.read('./config.ini')
    lookup_links = config.get('main', 'url_lookup')
    lookup_links = lookup_links.split(',')
    lookup_link = lookup_links[index]
    lookup_link = 'https://' + lookup_link + '/' 
    return lookup_link

def get_driver():
    """ returns selenium driver """
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--window-size=1420,1080')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-dev-shm-usage')        
    driver = webdriver.Chrome(chrome_options=chrome_options)
    return driver

def get_page_link(driver,search_url,attempts:int=3):
    """ get dois from page  """
    results = []
    for attempt in range(attempts):
        try:
            driver.get(search_url)
            referrences = driver.find_elements_by_class_name('article--citation')
            for ref in referrences:
                result = ref.text
                results.append(result)
            return results
        except Exception as e:
            print(str(e))
            driver.refresh()
            sleep(2)
    return 'Error'

def cycle_pages(driver,search_limit:int,url:str,attempts:int=3):
    """ cycle though result pages"""
    results = []
    for i in range(search_limit-1):
        search_url = url + str(i)
        result = get_page_link(driver,search_url)
        if result == 'Error': break
        results.append(result)
    return results

def aggregate_list(start_list):
    final_list = []
    for list_element in start_list:
        if len(list_element) > 0:
            for element in list_element:
                final_list.append(element)
    return final_list

def parse_doi(txt_source):
    """ get the doi number """
    try:
        doi = re.search(r'(10[.].*[0-9])', txt_source)
        doi = doi.group(0)
    except Exception as e:
        print(str(e))
        doi = 'error'
    return doi 

    # try: 
    #     pdf = PyPDF2.PdfFileReader(file_name)
    #     info = pdf.getDocumentInfo()
    #     result = {**info}
    #     pages = int(pdf.getNumPages())
    #     for page in range(pages):
    #         content = content + str(pdf.getPage(page).extractText().replace('\n',''))
    #     result['content'] = content  
    # except Exception as e:
    #     print(str(e))

    # return result 


def main():

    logging.info('doi script script has started ')

    # get selenium driver 
    driver  = get_driver()

    # read specific config 
    url, search_limit, search_term = read_config('jamanetwork')
    
    # get doi links 
    links = cycle_pages(driver,search_limit,url)

    # aggregate lists 
    final_list = aggregate_list(links)

    # create df for links 
    links_to_dois_df = pd.DataFrame(final_list,columns=['ref'])

    # get dois 
    links_to_dois_df['doi'] = links_to_dois_df['ref'].apply(lambda x: parse_doi(x))

    # save links to DB  
    #links_to_dois_df.to_sql('dois', conn, if_exists='append', index=False)
    links_to_dois_df['doi'].to_csv('/result/dois.csv')
    logging.info('saved to CSV /result/dois.csv')

if __name__ == "__main__":
    main()
