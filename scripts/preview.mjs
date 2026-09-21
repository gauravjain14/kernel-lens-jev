import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
const port = Number(process.env.PORT || 4173);
const allowed = new Map([['/panel.css', 'text/css'], ['/panel.js', 'text/javascript'], ['/preview.js', 'text/javascript']]);
createServer(async (request, response) => {
  const path = new URL(request.url || '/', 'http://localhost').pathname;
  try {
    if (path === '/') {
      let html = await readFile('media/panel.html', 'utf8');
      html = html.replaceAll('__CSP__', "'self'").replaceAll('__NONCE__', 'preview-only')
        .replaceAll('__STYLE__', '/panel.css').replaceAll('__SCRIPT__', '/panel.js')
        .replace('<script nonce="preview-only"', '<script nonce="preview-only" src="/preview.js"></script><script nonce="preview-only"');
      response.writeHead(200, { 'Content-Type': 'text/html', 'Cache-Control': 'no-store' }); response.end(html);
    } else if (allowed.has(path)) {
      response.writeHead(200, { 'Content-Type': allowed.get(path), 'Cache-Control': 'no-store' });
      const content = await readFile(`media${path}`, 'utf8');
      response.end(path === '/preview.js' ? content.replace('__SYSTEMS_SAMPLE__', await readFile('artifacts/systems-preview.json', 'utf8')) : content);
    } else { response.writeHead(404); response.end('Not found'); }
  } catch { response.writeHead(500); response.end('Preview unavailable'); }
}).listen(port, '127.0.0.1', () => console.log(`Kernel Lens interface preview: http://127.0.0.1:${port} (sample data only)`));
