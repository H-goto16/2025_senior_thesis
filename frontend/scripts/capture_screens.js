// Capture key screens of the Expo web app using Playwright (chromium)
const { chromium } = require('@playwright/test');
const path = require('path');
const fs = require('fs');

async function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

async function run() {
  const outDir = path.resolve(__dirname, '../../LaTex/Image_Goto');
  await ensureDir(outDir);

  const baseUrl = process.env.APP_BASE_URL || 'http://localhost:8081';
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } }); // iPhone-ish size
  const page = await context.newPage();

  // Helper to navigate and capture
  async function snap(route, name) {
    await page.goto(`${baseUrl}${route}`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(800);
    await page.screenshot({ path: path.join(outDir, `${name}.png`), fullPage: true });
  }

  // Assumes Expo Router tabs exist at these routes
  await snap('/', 'home');
  await snap('/(tabs)/detection', 'Detection_tab');
  await snap('/(tabs)/labeling', 'Labeling_tab');
  await snap('/(tabs)/training', 'Training_tab');
  await snap('/(tabs)/models', 'Models_tab');
  await snap('/(tabs)/analytics', 'Analytics_tab');

  await browser.close();
  console.log('Screens captured to:', outDir);
}

run().catch((e) => {
  console.error(e);
  process.exit(1);
});




