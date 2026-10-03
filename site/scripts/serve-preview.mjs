import { readFile, stat } from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';

const rootArg = process.argv[2];
if (!rootArg) {
  console.error('Usage: node site/scripts/serve-preview.mjs <built-site-directory> [port]');
  process.exit(2);
}
const root = path.resolve(rootArg);
const port = Number(process.argv[3] ?? 4173);
const prefix = '/social-deduction-ai/';
const mime = new Map([
  ['.css', 'text/css; charset=utf-8'], ['.html', 'text/html; charset=utf-8'], ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'], ['.png', 'image/png'], ['.webp', 'image/webp']
]);
const isWithin = (parent, child) => {
  const relative = path.relative(parent, child);
  return relative === '' || (relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative));
};

if (!(await stat(root)).isDirectory() || !Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error('Provide an existing built-site directory and a valid TCP port.');
}

http.createServer(async (request, response) => {
  const pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
  if (!pathname.startsWith(prefix)) {
    response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }).end('Not found');
    return;
  }
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    response.writeHead(405, { Allow: 'GET, HEAD' }).end();
    return;
  }
  let relative;
  try { relative = decodeURIComponent(pathname.slice(prefix.length)); } catch {
    response.writeHead(400).end('Bad path');
    return;
  }
  let file = path.resolve(root, relative || 'index.html');
  if (!isWithin(root, file)) {
    response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }).end('Not found');
    return;
  }
  try {
    if ((await stat(file)).isDirectory()) file = path.join(file, 'index.html');
    const body = await readFile(file);
    response.writeHead(200, {
      'Cache-Control': 'no-store',
      'Content-Length': body.length,
      'Content-Type': mime.get(path.extname(file).toLowerCase()) ?? 'application/octet-stream'
    });
    if (request.method === 'HEAD') response.end(); else response.end(body);
  } catch {
    response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }).end('Not found');
  }
}).listen(port, '127.0.0.1', () => {
  console.log(`Preview at http://127.0.0.1:${port}${prefix}`);
});
