// Offline provenance/request verification and replay of the actual captured answer.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { BlockReader } = require('../../src/core/blocks.ts');
const { analyzeSystems } = require('../../src/core/systems/analyze.ts');
const { systemsPayload, systemsReport } = require('../../src/core/systems/evaluate.ts');
const { parseAnswers } = require('../../src/core/gateway.ts');

const root = path.resolve(__dirname, '../..');
const read = file => fs.readFileSync(path.join(__dirname, file));
const json = file => JSON.parse(read(file));
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const provenance = json('captures/provenance.json');
for (const [file, expected] of Object.entries(provenance.engine.filesSha256)) {
  assert.equal(sha(fs.readFileSync(path.join(root, file))), expected, `Engine changed: ${file}`);
}
for (const [variant, record] of Object.entries(provenance.variants)) {
  for (const [file, expected] of Object.entries(record.sha256)) {
    assert.equal(sha(read(file)), expected, `Capture changed: ${file}`);
  }
  const source = read(`${variant}.py`).toString('utf8');
  const helpers = read(`${variant}_utils.py`).toString('utf8').split('\n');
  const helperLine = helpers.findIndex(line => /^def _map_moe_params_(?:common|qwen3_moe)\(/.test(line));
  assert(helperLine >= 0);
  const references = [{ name: 'MOE_PARAM_HANDERS', startLine: helperLine + 1,
    code: helpers.slice(helperLine).join('\n'), reason: 'Imported source dependency from the same upstream revision' }];
  const context = new BlockReader().read({ source, file: 'transformer_impl.py', language: 'python',
    cursorLine: source.split('\n').findIndex(line => line.includes('def get_per_tensor_param(')),
    assessmentScope: provenance.settings.assessmentScope,
    hardwareProfile: provenance.settings.hardwareProfile, hardware: provenance.settings.hardware, references });
  const analysis = analyzeSystems(context);
  const payload = systemsPayload(context, analysis);
  assert.equal(JSON.stringify(payload), read(`captures/${variant}.request.json`).toString('utf8'), 'Request differs');
  if (record.status !== 'captured') {
    console.log(`${variant}: request verified; Gateway HTTP ${record.httpStatus}, no classification available.`);
    continue;
  }
  const response = json(`captures/${variant}.response.json`);
  const answers = parseAnswers(response, payload);
  const report = systemsReport(context, analysis, answers, record.usage, provenance.settings.threshold);
  assert.deepEqual(JSON.parse(JSON.stringify(report)), json(`captures/${variant}.report.json`), 'Report differs');
  for (const finding of report.findings) {
    console.log(`${variant}: ${finding.title}; ${finding.assessment}; ${Math.round(finding.model_concern_probability * 100)}% concern mass`);
  }
}
console.log('Verified frozen source, engine hashes and request bytes. No network or training execution.');
