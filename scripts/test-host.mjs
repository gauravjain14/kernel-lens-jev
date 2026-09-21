import { build } from 'esbuild';
import { mkdtemp, mkdir, readFile } from 'node:fs/promises';
import { parseEnv } from 'node:util';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawn, execFileSync } from 'node:child_process';
const scenarioArg = process.argv.indexOf('--scenario');
const scenario = scenarioArg >= 0 ? process.argv[scenarioArg + 1] : 'all';
if (!['all', 'systems', 'legacy', 'reduction', 'workflows', 'release', 'cuda-asm'].includes(scenario)) throw new Error('Unknown editor test scenario.');
const testFile = scenario === 'cuda-asm' ? 'test/cuda-asm.host.ts' : scenario === 'release' ? 'test/release.host.ts' : scenario === 'legacy' ? 'test/host.integration.ts' : 'test/systems.host.ts';
await build({ entryPoints: [testFile], outfile: 'dist/host-tests.cjs', bundle: true,
  platform: 'node', target: 'node20', format: 'cjs', external: ['vscode'] });
const profile = await mkdtemp(path.join(tmpdir(), 'kernel-lens-host-'));
await mkdir(`${profile}/workspace`);
const vsixArg = process.argv.indexOf('--vsix');
let extensionPath = process.cwd();
if (vsixArg >= 0) {
  const vsix = process.argv[vsixArg + 1];
  if (!vsix) throw new Error('--vsix requires an artifact path');
  execFileSync('python', ['-c', 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])', path.resolve(vsix), `${profile}/package`]);
  extensionPath = `${profile}/package/extension`;
  console.log(`Testing packaged VSIX: ${path.basename(vsix)}`);
}
const binary = process.env.VSCODE_EXECUTABLE || (process.platform === 'linux' ? '/usr/share/code/code' : 'code');
const args = ['--no-sandbox', '--disable-gpu', '--disable-workspace-trust', '--skip-welcome', '--skip-release-notes', '--disable-telemetry',
  `--user-data-dir=${profile}/profile`, `--extensions-dir=${profile}/extensions`,
  `--extensionDevelopmentPath=${extensionPath}`, `--extensionTestsPath=${process.cwd()}/dist/host-tests.cjs`, `${profile}/workspace`];
const env = { ...process.env };
const envFileArg = process.argv.indexOf('--env-file');
if (envFileArg >= 0) env.AI_GATEWAY_API_KEY = parseEnv(await readFile(process.argv[envFileArg + 1], 'utf8')).AI_GATEWAY_API_KEY;
env.KERNEL_LENS_TEST_ROOT = process.cwd();
env.KERNEL_LENS_CAPTURE = process.argv.includes('--capture') ? '1' : '0';
env.KERNEL_LENS_LIVE = process.argv.includes('--live') ? '1' : '0';
const workflowArg = process.argv.indexOf('--workflow');
env.KERNEL_LENS_WORKFLOW_FILTER = workflowArg >= 0 ? process.argv[workflowArg + 1] ?? '' : '';
delete env.ELECTRON_RUN_AS_NODE;
if (env.KERNEL_LENS_LIVE !== '1') delete env.AI_GATEWAY_API_KEY;
const child = spawn(process.platform === 'linux' ? 'xvfb-run' : binary,
  process.platform === 'linux' ? ['-a', binary, ...args] : args, { env, stdio: 'inherit' });
const timeout = setTimeout(() => { child.kill('SIGTERM'); console.error('Extension-host test timed out.'); }, scenario === 'workflows' ? 360000 : 120000);
child.on('error', error => { clearTimeout(timeout); console.error(error.message); process.exitCode = 1; });
child.on('exit', code => { clearTimeout(timeout); process.exitCode = code ?? 1; });
