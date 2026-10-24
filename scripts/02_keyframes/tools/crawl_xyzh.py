"""Real old photos from the ZAFU alumni association site (校友总会): 母校印记 / 毕业纪念册 / 校友园地 / 校友刊物.
  .venv/bin/python tools/crawl_xyzh.py  -> ref/web/old/xyzh/<section>/<art>_<k>.jpg + index.json (article title, date, page, caption)
"""
import re, json, time, hashlib
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'ref/web/old/xyzh'
BASE = 'https://xyzh.zafu.edu.cn/'
UA = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'}
SECTIONS = ['xyh/mxyj', 'xyh/byjnc', 'xyh/xyyd', 'xyh/xykw']
S = requests.Session(); S.headers.update(UA)


def get(url):
    for _ in range(3):
        try:
            r = S.get(url, timeout=30)
            r.encoding = 'utf-8'
            return r
        except Exception:
            time.sleep(2)
    return None


def list_articles(sec):
    arts, pages = {}, [BASE + sec + '.htm']
    r = get(pages[0])
    if r is None:
        return arts
    m = re.search(r'共\s*(\d+)\s*页|/(\d+)\s*</', r.text)
    n = 1
    for mm in re.finditer(r'href="' + re.escape(sec.split('/')[-1]) + r'/(\d+)\.htm"', r.text):
        n = max(n, int(mm.group(1)))
    for k in range(1, n + 1):
        pages.append(BASE + sec + f'/{k}.htm')
    for p in pages:
        rr = get(p)
        if rr is None or rr.status_code != 200:
            continue
        for mm in re.finditer(r'href="([^"]*info/\d+/\d+\.htm)"[^>]*>(.*?)</a>', rr.text, re.S):
            u = urljoin(p, mm.group(1)); t = re.sub(r'<[^>]+>', '', mm.group(2)).strip()
            if u not in arts or (t and not arts[u]):
                arts[u] = t
    return arts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx_p = OUT / 'index.json'
    index = json.loads(idx_p.read_text()) if idx_p.exists() else []
    done = {x['page'] for x in index}
    seen = {x['sha1'] for x in index}
    for sec in SECTIONS:
        arts = list_articles(sec)
        print(sec, len(arts), 'articles', flush=True)
        d = OUT / sec.split('/')[-1]; d.mkdir(exist_ok=True)
        for u, t in arts.items():
            if u in done:
                continue
            r = get(u)
            if r is None or r.status_code != 200:
                continue
            title = t or (re.search(r'<title>(.*?)</title>', r.text, re.S).group(1).strip() if '<title>' in r.text else '')
            date = (re.search(r'(20\d\d|19\d\d)[-年.](\d\d?)[-月.](\d\d?)', r.text) or [None])[0]
            body = r.text.split('vsbcontent')[-1] if 'vsbcontent' in r.text else r.text
            for k, mm in enumerate(re.finditer(r'<img[^>]+src="([^"]+__local[^"]+)"[^>]*>', body)):
                iu = urljoin(u, mm.group(1))
                try:
                    data = S.get(iu, timeout=40).content
                    im = Image.open(BytesIO(data)); im.load()
                except Exception:
                    continue
                if min(im.size) < 400:
                    continue
                h = hashlib.sha1(data).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                art = u.rstrip('.htm').split('/')[-1]
                fn = d / f'{art}_{k:02d}.jpg'
                if im.format == 'JPEG':
                    fn.write_bytes(data)
                else:
                    im.convert('RGB').save(fn, quality=93)
                # caption: text right after the image (first short paragraph)
                after = re.sub(r'<[^>]+>', ' ', body[mm.end(): mm.end() + 600])
                cap = re.sub(r'\s+', ' ', after).strip()[:80]
                index.append({'section': sec, 'page': u, 'title': title, 'date': date, 'img': iu, 'file': str(fn.relative_to(ROOT)),
                              'size': im.size, 'sha1': h, 'caption': cap})
            done.add(u)
            idx_p.write_text(json.dumps(index, ensure_ascii=False, indent=1))
        print(sec, 'images so far', len(index), flush=True)


if __name__ == '__main__':
    main()
