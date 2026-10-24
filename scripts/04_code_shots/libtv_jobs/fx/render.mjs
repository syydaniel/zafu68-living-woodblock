// Render an fx page in headless Chrome and save raw RGBA frames.
//   node render.mjs <fx> [--w 1920 --h 1080 --K 8 --shutter 0.5 --list 0,5,10 --out out/<fx>/raw --headed]
// Files are served from this folder through request interception (no local server needed).
import { chromium } from '../../node_modules/playwright-core/index.mjs';
import fs from 'node:fs';
import path from 'node:path';

const argv = process.argv.slice(2);
const fx = argv[0];
const opt = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? (argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : true) : d; };
const W = +opt('w', 1920), H = +opt('h', 1080), K = +opt('K', 8), SH = +opt('shutter', 0.5);
const ROOT = path.dirname(new URL(import.meta.url).pathname);
const out = path.join(ROOT, opt('out', `out/${fx}/raw`));
fs.mkdirSync(out, { recursive: true });

const browser = await chromium.launch({
  channel: 'chrome', headless: !opt('headed', false),
  args: ['--ignore-gpu-blocklist', '--use-angle=metal', '--enable-gpu', '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding', '--js-flags=--max-old-space-size=8192'],
});
const page = await browser.newPage({ viewport: { width: Math.min(W, 1920), height: Math.min(H, 1080) } });
page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') console.error('PAGE', m.type(), m.text().slice(0, 300)); });
page.on('pageerror', e => console.error('PAGEERR', e.message));
const MIME = { '.js': 'text/javascript', '.html': 'text/html', '.png': 'image/png', '.jpg': 'image/jpeg', '.json': 'application/json' };
await page.route('http://fx.local/**', route => {
  const p = path.join(ROOT, decodeURIComponent(new URL(route.request().url()).pathname));
  if (!p.startsWith(ROOT) || !fs.existsSync(p)) return route.fulfill({ status: 404, body: 'nf' });
  route.fulfill({ status: 200, body: fs.readFileSync(p), headers: { 'content-type': MIME[path.extname(p)] || 'application/octet-stream' } });
});
await page.goto(`http://fx.local/page/index.html?fx=${fx}&w=${W}&h=${H}&K=${K}&shutter=${SH}`);
await page.waitForFunction('window.READY === true || window.BOOT_ERROR', null, { timeout: 300000 });
const err = await page.evaluate('window.BOOT_ERROR');
if (err) { console.error('BOOT', err); await browser.close(); process.exit(1); }
const N = await page.evaluate('window.FRAMES');
const list = opt('list', null) ? String(opt('list')).split(',').map(Number) : [...Array(N).keys()];
const t0 = Date.now();
for (const i of list) {
  const b64 = await page.evaluate(i => window.renderFrame(i), i);
  fs.writeFileSync(path.join(out, String(i).padStart(5, '0') + '.rgba'), Buffer.from(b64, 'base64'));
}
console.log(`DONE ${fx} ${list.length} frames ${W}x${H} K=${K} in ${((Date.now() - t0) / 1000).toFixed(1)} s (N=${N})`);
await browser.close();
