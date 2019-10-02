#############################################################
# Deep Guitar Data Scraper
#############################################################

import urllib3
from bs4 import BeautifulSoup
import requests
import warnings

warnings.filterwarnings("ignore")

domain = 'https://www.classicalguitarmidi.com/'


# TODO: Change path based on where you want the data stored!!

path = '/Users/aditya/Documents/Data_Science/datasets/deepguitar-data/'

urls = ['https://www.classicalguitarmidi.com/a-b.html',
		'https://www.classicalguitarmidi.com/c-f.html',
		'https://www.classicalguitarmidi.com/g-l.html', 
		'https://www.classicalguitarmidi.com/m-o.html',
		'https://www.classicalguitarmidi.com/p-r.html',
		'https://www.classicalguitarmidi.com/s-z.html',
		'https://www.classicalguitarmidi.com/bach.html',
		'https://www.classicalguitarmidi.com/barrios.html',
		'https://www.classicalguitarmidi.com/coste.html', 
		'https://www.classicalguitarmidi.com/giuliani.html',
		'https://www.classicalguitarmidi.com/sor.html',
		'https://www.classicalguitarmidi.com/tarrega.html'] 


#############################################################
# HELPER FUNCTIONS
#############################################################


def scrape_filenames(url):
	page = requests.get(url, verify=False)
	soup = BeautifulSoup(page.text, 'html.parser')
	a_vals = soup.find_all('a')
	filenames = []
	for i in range(len(a_vals)):
		link = a_vals[i]
		href = str(link.get('href'))
		if ('subivic/' in href):
			filenames.append(href)
	return filenames

def download_all():
	files = []
	for url in urls:
		files += scrape_filenames(url)
	for file in files:
		r = requests.get(domain+file, verify=False)
		with open(path+file[8:], 'wb') as f:
			f.write(r.content)
		

#############################################################
# CORE CODE
#############################################################


download_all()
