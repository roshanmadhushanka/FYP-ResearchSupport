from bs4 import BeautifulSoup
from urllib.request import urlopen
from bs4.element import Tag


def traverse(soup, path="", maps={}):
    if soup.name is not None:
        maps[path] = soup.text
        if isinstance(soup, Tag):
            count = 0
            for child in soup.children:
                count += 1
                traverse(child, path + ">" + str(child.name) + str(count), maps)


page = urlopen("https://en.wikipedia.org/wiki/Russian_Ground_Forces")
soup = BeautifulSoup(page, 'html.parser')

maps = {}
traverse(soup, maps=maps)

f = open('version5.html', mode='w', encoding='utf-8')
f.write(soup.prettify())
f.close()

