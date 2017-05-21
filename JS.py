from bs4 import BeautifulSoup
from urllib.request import urlopen

print("Opening URL")
page = urlopen("https://www.youtube.com/watch?v=sYckquWBxjo&index=2&list=PLC3y8-rFHvwg5gEu2KF4sbGvpUqMRSBSW")

print("Convert to soup")
soup = BeautifulSoup(page, "lxml")

print("Tags")
li_tags = soup.findAll(name='li', attrs={'class': 'yt-uix-scroller-scroll-unit'})
for li in li_tags:
    a_tags = li.findAll(name='a')
    for a in a_tags:
        print(a)