from bs4 import BeautifulSoup
import requests
import json
import matplotlib.pyplot as plt

#resp = requests.get('http://finance.yahoo.com/echarts?s=JNPR#{"range":"max","allowChartStacking":true}')
url = "http://finance.google.com/finance/info?client=ig&q=NASDAQ:NVDA"

resp = requests.get(url=url)
soup =  BeautifulSoup(resp.text, 'html.parser')

json_data = soup.get_text()#.find('var ')

print 'json data is:', json_data[3:]

stock_obj = json.loads(json_data[3:])
stockdetails = dict(stock_obj[0])
print stockdetails['el']
