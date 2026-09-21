import { build, context } from 'esbuild';
import { mkdir } from 'node:fs/promises';
await mkdir('artifacts', { recursive: true });
const options = {
  entryPoints: ['src/extension.ts'], outfile: 'dist/extension.cjs', bundle: true,
  platform: 'node', target: 'node20', format: 'cjs', external: ['vscode'], sourcemap: true,
};
if (process.argv.includes('--watch')) {
  const ctx = await context(options);
  await ctx.watch();
  console.log('Watching Kernel Lens. Press F5 in VS Code.');
} else {
  await build(options);
  console.log('Built dist/extension.cjs');
}
