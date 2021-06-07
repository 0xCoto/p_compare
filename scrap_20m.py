from bs4 import BeautifulSoup, SoupStrainer
import requests

url = "https://www.gb.nrao.edu/20m/peak/log2013.htm"

page = requests.get(url)
data = page.text
soup = BeautifulSoup(data)

for link in soup.find_all('a'):
	if link.get('href') is not None:
		print(link.get('href'))
		if '.psr.' in link.get('href'):
			print(link.get('href'))
			presto_link = link.get('href').replace('.psr.htm', '_psr.jpg')
			img_data = requests.get('https://www.gb.nrao.edu/20m/peak/'+presto_link).content
			with open(presto_link.split('/')[1], 'wb') as handler:
				handler.write(img_data)
