# imports libaries
import requests						# HTTP connection
import time							# timer
import pickle						# exports lists
import numpy as np					# numerical computation
from bs4 import BeautifulSoup		# web scraping

# generates random sample of users
n = 1000
max_id = 12600000
storyids = np.random.randint(1, max_id, n)
storyids = np.unique(storyids)
storyids = [str(storyid) for storyid in storyids]
baselink = 'https://www.fanfiction.net/s/'
urls = [baselink + storyid for storyid in storyids]

# initializes datasets
data_stories = []

# collects data from user pages
t0 = time.time()
for i in range(0, n, 1):
    
    # retrieves user identifier
    storyid = storyids[i]

    # collects data from webpage
    page = ''
    while page == '':
        try:
            page = requests.get(urls[i])
        except:
            print("Connection refused, going to sleep...")
            time.sleep(5)
            continue
            
    html = page.text
    soup = BeautifulSoup(html, 'html.parser')
    
    # sets default for missing stories
    userid = 'NA'
    cat = 'NA'
    title = 'NA'
    summary = 'NA'
    info = 'NA'
    text = 'NA'
    error = 'NA'
 
    # collects story information if story exists
    if soup.find('span', {'class': 'gui_warning'}) is None:
        useridtag = soup.find('a', {'title': 'Send Private Message'})
        cattag = soup.find('div', {'id': 'pre_story_links'})
        titletag = soup.find('b', {'class': 'xcontrast_txt'})
        summarytag = soup.find('div', {'class': 'xcontrast_txt',
                                       'style' : 'margin-top:2px'})
        infotag = soup.find('span', {'class': 'xgray xcontrast_txt'})
        texttag = soup.find('div', {'id': 'storytext'})
        
        if useridtag is None:
            error = soup.find('span').text
        else:
            userid = useridtag['href']
            cat = [link.text for link in cattag.find_all('a')]
            title = titletag.text
            summary = summarytag.text
            info = infotag.text
            text = texttag.text
        
        story = [storyid, userid, cat, title, summary, info, text, error]
        data_stories.append(story)
    
time.time() - t0

# exports data
with open('data/raw_data/data_stories_text', 'wb') as fp:
    pickle.dump(data_stories, fp)