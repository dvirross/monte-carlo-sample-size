// Render each figures/*.json to a static PNG (2200 px wide) with plotly.js in headless Chromium.
// Usage: node scripts/render_png.mjs <path to plotly.min.js>
// The PNGs are the static fallbacks shown before the interactive chart loads, in print and without JavaScript.
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
const { chromium } = createRequire(import.meta.url)('playwright');

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const plotlyPath = process.argv[2];
if (!plotlyPath) throw new Error('pass the path to a plotly.js bundle');
const plotly = readFileSync(plotlyPath, 'utf8');
const WIDTH = 1100; // CSS pixels; deviceScaleFactor 2 gives 2200 px

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: WIDTH, height: 900 }, deviceScaleFactor: 2 });
await page.setContent('<html><body style="margin:0;background:#fff"><div id="g"></div></body></html>');
await page.addScriptTag({ content: plotly });
for (const f of readdirSync(join(root, 'figures')).filter((n) => n.endsWith('.json')).sort()) {
  const spec = JSON.parse(readFileSync(join(root, 'figures', f), 'utf8'));
  const height = spec.layout.height ?? 500;
  await page.setViewportSize({ width: WIDTH, height });
  await page.evaluate(async ({ spec, WIDTH, height }) => {
    const el = document.getElementById('g');
    await window.Plotly.newPlot(el, spec.data, { ...spec.layout, width: WIDTH, height }, { staticPlot: true });
  }, { spec, WIDTH, height });
  const png = await page.locator('#g').screenshot();
  writeFileSync(join(root, 'figures', f.replace('.json', '.png')), png);
  console.log('rendered', f, `${WIDTH * 2}x${height * 2}`);
}
await browser.close();
