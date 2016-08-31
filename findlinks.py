# connects to server
import urllib
url = ""
html = urllib.urlopen(url).read()

# collects data from webpage
from BeautifulSoup import *
soup = BeautifulSoup(html)

# stores all link data in list
links = []
tags = soup('a')
for tag in tags:
  links.append(tag.get('href'))
  
# imports regular expression
import re

# stores user and story links
userlinks = []
storylinks = []
user = re.compile('/u/')
story = re.compile('/s/')
for link in links:
  text = str(link)
  if user.search(text) is not None:
    u = re.search('/u/(.+?)/', text).group(1)
    userlinks.append(u)
  if story.search(text) is not None:
    s = re.search('/s/(.+?)/', text).group(1)
    storylinks.append(s)

linksoflinks = []
linksoflinks.append(userlinks)
linksoflinks.append(storylinks)

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