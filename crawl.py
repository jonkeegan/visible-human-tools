import re, sys, urllib.request, urllib.parse, concurrent.futures as cf
BASE="https://data.lhncbc.nlm.nih.gov/public/Visible-Human/"
row=re.compile(r"<a href='([^']+)'>[^<]*</a></td><td class='size'>([^<]*)</td>")
files=[]
def fetch(path):
    url=BASE+urllib.parse.quote(path)+"index.html"
    html=urllib.request.urlopen(url,timeout=60).read().decode()
    subs=[]
    for href,size in row.findall(html):
        href=urllib.parse.unquote(href)
        if href.endswith('/index.html'): subs.append(path+href[:-10])
        else: files.append((path+href,int(size)))
    return subs
todo=["Male-Images/","Female-Images/"]
with cf.ThreadPoolExecutor(16) as ex:
    while todo:
        res=list(ex.map(fetch,todo)); todo=[s for r in res for s in r]
with open("files.tsv","w") as f:
    for p,s in sorted(files): f.write(f"{s}\t{p}\n")
