"""Reference photo gathering: Bing image search -> ref/web/bing/<slug>/NN.jpg + index.json (query, source page, image url).
  .venv/bin/python tools/bing_images.py [--n 30]
Real photos only (official site first, then press/blogs); they are references for the image-model reconstruction, never faked.
"""
import json, re, sys, hashlib, html, urllib.parse, concurrent.futures as cf
from io import BytesIO
from pathlib import Path
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'ref/web/bing'
UA = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'}
N = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 30
QUERIES = {
    'lib': '浙江农林大学 东湖校区 图书馆',
    'lib_reflect': '浙江农林大学 图书馆 东湖 倒影',
    'donghu': '浙江农林大学 东湖',
    'aerial': '浙江农林大学 航拍',
    'panorama': '浙江农林大学 全景 鸟瞰',
    'gate': '浙江农林大学 校门',
    'ginkgo': '浙江农林大学 银杏',
    'yijin': '浙江农林大学 衣锦校区',
    'botanic': '浙江农林大学 植物园',
    'motto': '浙江农林大学 求真敬业 校训',
    'metasequoia': '浙江农林大学 水杉',
    'bamboo': '浙江农林大学 竹林',
    'snow': '浙江农林大学 雪景',
    'night': '浙江农林大学 夜景',
    'autumn': '浙江农林大学 秋色',
    'spring': '浙江农林大学 春天 花',
    'lab': '浙江农林大学 实验室 科研',
    'field': '浙江农林大学 学生 野外实习',
    'tianmu': '天目山 国家级自然保护区',
    'tianmu_trees': '天目山 大树王 柳杉',
    'qianwang': '临安 钱王陵',
    'qingshanhu': '临安 青山湖',
    'old1': '浙江农林大学 老照片',
    'old2': '浙江林学院 老照片',
    'old3': '天目林学院',
    'history': '浙江农林大学 校史馆',
    'admin': '浙江农林大学 行政楼',
    'graduation': '浙江农林大学 毕业典礼',
    'seed': '银杏种子 白果 特写 摄影',
    'seedling': '银杏 幼苗 发芽 子叶',
    'fruit': '银杏果 枝头 金黄 树叶',
    'specimen': '浙江农林大学 植物标本馆',
    'herbarium': '腊叶标本 银杏',
    'zwmstk': '植物名实图考 银杏',
    'roots': '大树 根系 土壤 特写',
    'starry': '浙江农林大学 星空',
    'oldtree': '浙江农林大学 古树 银杏 衣锦',
}


def search(q):
    urls, seen = [], set()
    for first in (1, 36, 71):
        r = requests.get('https://www.bing.com/images/search', params={'q': q, 'form': 'HDRSC2', 'first': first}, headers=UA, timeout=30)
        for m in re.finditer(r'm="(\{[^"]+\})"', r.text):
            try:
                d = json.loads(html.unescape(m.group(1)))
            except Exception:
                continue
            u = d.get('murl')
            if u and u not in seen:
                seen.add(u); urls.append({'img': u, 'page': d.get('purl', ''), 'title': d.get('t', '')})
        if len(urls) >= N * 2:
            break
    return urls


def exif_date(im):
    try:
        ex = im.getexif()
        return str(ex.get_ifd(0x8769).get(36867) or ex.get(306) or '')
    except Exception:
        return ''


def fetch(item):
    try:
        r = requests.get(item['img'], headers=UA, timeout=40)
        if r.status_code != 200 or len(r.content) < 30000:
            return None
        im = Image.open(BytesIO(r.content)); im.load()
        if min(im.size) < 700:
            return None
        return r.content, im.size
    except Exception:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx_path = OUT / 'index.json'
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    hashes = {v['sha1'] for vs in index.values() for v in vs}
    for slug, q in QUERIES.items():
        if slug in index and len(index[slug]) >= min(N, 8):
            continue
        d = OUT / slug; d.mkdir(exist_ok=True)
        cands = search(q)
        got = []
        with cf.ThreadPoolExecutor(8) as ex:
            for item, res in zip(cands, ex.map(fetch, cands)):
                if res is None or len(got) >= N:
                    continue
                data, size = res
                h = hashlib.sha1(data).hexdigest()
                if h in hashes:
                    continue
                hashes.add(h)
                fn = d / f'{len(got) + 1:02d}.jpg'
                im = Image.open(BytesIO(data))
                if im.format == 'JPEG':
                    fn.write_bytes(data)            # keep the original file (EXIF capture date matters for history)
                else:
                    im.convert('RGB').save(fn, quality=93)
                got.append({**item, 'file': str(fn.relative_to(ROOT)), 'size': size, 'sha1': h, 'date': exif_date(im)})
        index[slug] = got
        idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=1))
        print(f'{slug:12s} {q}: {len(cands)} found, {len(got)} kept', flush=True)


if __name__ == '__main__':
    main()
