# imports libraries
import urllib						# server connection
import re 							# regular expression
import numpy as np 					# numerical computation
from bs4 import BeautifulSoup		# web scraping

# generate random sample of users
n = 100
max_id = 9000000
userids = np.random.randint(1, max_id, n)
userids = [str(userid) for userid in userids]
baselink = 'https://www.fanfiction.net/u/'
urls = [baselink + userid for userid in userids]

# initializes datasets
user_data = []
story_data = []

for i in range(0, n, 1):
    # connects to server
    url = urls[i]
    html = urllib.request.urlopen(url).read()
    
    # collects data from webpage
    soup = BeautifulSoup(html, 'html.parser')
    
    # stores all link data in list
    links = []
    tags = soup('a')
    for tag in tags:
      links.append(tag.get('href'))
      
    # stores user links
    userlinks = []
    user = re.compile('/u/')
    for link in links:
      text = str(link)
      if user.search(text) is not None:
        u = re.search('/u/(.+?)/', text).group(1)
        userlinks.append(u)
    
    # stores story links
    storylinks = []
    story = re.compile('/s/')
    for link in links:
      text = str(link)
      if story.search(text) is not None:
        s = re.search('/s/(.+?)/', text).group(1)
        storylinks.append(s)
    
    user_data.append(userlinks)
    story_data.append(storylinks)
    
# writes file
file = open('test.txt','w')
for item in storylinks:
  file.write('%s\n' % item)

# writes table
import csv
import sys

with open('test2.csv','w') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerows(linksoflinks)