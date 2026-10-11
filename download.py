# Resumable parallel downloader for the Visible Human dataset.
# Re-run anytime: files already present with the correct size are skipped.
# Usage: python3 download.py [threads] [list.tsv]   (list defaults to files.tsv)
import os, sys, time, urllib.request, urllib.parse, shutil, concurrent.futures as cf, threading
BASE="https://data.lhncbc.nlm.nih.gov/public/Visible-Human/"
ROOT=os.path.dirname(os.path.abspath(__file__))
jobs=[(int(s),p) for s,p in (l.rstrip("\n").split("\t",1) for l in open(os.path.join(ROOT,sys.argv[2] if len(sys.argv)>2 else "files.tsv")))]
total=sum(s for s,_ in jobs); lock=threading.Lock(); done=[0,0,0]  # bytes, files, failures
def get(job):
    size,path=job; dest=os.path.join(ROOT,path)
    if os.path.exists(dest) and os.path.getsize(dest)==size:
        with lock: done[0]+=size; done[1]+=1
        return
    os.makedirs(os.path.dirname(dest),exist_ok=True)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(BASE+urllib.parse.quote(path),timeout=120) as r, open(dest+".part","wb") as f:
                shutil.copyfileobj(r,f,1<<20)
            if os.path.getsize(dest+".part")!=size: raise IOError("size mismatch")
            os.replace(dest+".part",dest)
            with lock: done[0]+=size; done[1]+=1
            return
        except Exception as e:
            err=e; time.sleep(2**attempt)
    with lock: done[2]+=1
    print(f"FAILED {path}: {err}",flush=True)
def report():
    while True:
        time.sleep(60)
        with lock: print(f"{time.strftime('%H:%M:%S')} {done[1]}/{len(jobs)} files, {done[0]/1e9:.1f}/{total/1e9:.1f} GB, {done[2]} failed",flush=True)
threading.Thread(target=report,daemon=True).start()
with cf.ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv)>1 else 8) as ex: list(ex.map(get,jobs))
print(f"DONE {done[1]}/{len(jobs)} files, {done[0]/1e9:.1f} GB, {done[2]} failed",flush=True)
