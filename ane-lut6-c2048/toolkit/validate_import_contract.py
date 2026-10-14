#!/usr/bin/env python3
"""Exercise the production manifest builder against Swarmer's Swift importer.

Uses synthetic package payloads and the pinned source declaration, NOT source
weights. This is contract validation, not a full export or Core ML execution.
Outputs are retained for inspection; use a new output directory for each run.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import zipfile

from download_source import pinned_source_weight_identity
from run_export import build_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--swarmer', type=Path, required=True)
    parser.add_argument('--swiftc', default='swiftc')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    toolkit = Path(__file__).resolve().parent
    identity = pinned_source_weight_identity(json.loads((toolkit / 'source-verification.json').read_text()))
    blocks = [f'block_{i:02d}' for i in range(7)]
    parts = [('embedding', 'embedding', 0, 0)] + [
        (name, 'block', i * 4, (i + 1) * 4) for i, name in enumerate(blocks)
    ] + [('head', 'head', 0, 0)]
    manifest = build_manifest(SimpleNamespace(context=2048, bits=6), parts, blocks, identity)
    bundle = output / 'synthetic-models'
    bundle.mkdir()
    for name in manifest['model_files']:
        (bundle / name).mkdir()
        (bundle / name / 'NOT_A_MODEL.txt').write_text('Synthetic import contract fixture, not executable Core ML.\n')
    for name in ('rope_cos.fp16', 'rope_sin.fp16'):
        (bundle / name).write_bytes(bytes(2048 * 128 * 2))
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        (bundle / name).write_text('{}')
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Serialize, ZIP, extract, then consume using the actual Swift reader.
    archive_path = output / 'SYNTHETIC-contract.zip'
    with zipfile.ZipFile(archive_path, 'x') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(bundle))
    extracted = output / 'extracted'
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)  # locally generated, fixed member paths
    module = args.swarmer.resolve() / 'mobile/modules/swarmer-local-inference'
    binary = output / 'manifest-import-test'
    command = [args.swiftc, '-swift-version', '6', '-strict-concurrency=complete',
               '-warnings-as-errors', '-parse-as-library', '-module-cache-path', str(output / 'module-cache'),
               str(module / 'ios/DolphinCoreMLBundle.swift'),
               str(module / 'ios-tests/DolphinManifestImportTests.swift'), '-o', str(binary)]
    compilation = subprocess.run(command, capture_output=True, text=True)
    (output / 'compile.log').write_text(compilation.stdout + compilation.stderr)
    compilation.check_returncode()
    cases = [('schema3-roundtrip', manifest, True)]
    legacy = copy.deepcopy(manifest)
    legacy['format_version'] = 2
    for key in ('source_model', 'source_revision', 'source_weight_identity'):
        legacy.pop(key)
    cases.append(('historical-schema2', legacy, True))
    for label, key, value in [('future-schema', 'format_version', 4),
                              ('missing-identity', 'source_weight_identity', None),
                              ('wrong-source', 'source_model', 'untrusted/model'),
                              ('wrong-revision', 'source_revision', '0' * 40),
                              ('wrong-context', 'context_length', 4096)]:
        changed = copy.deepcopy(manifest)
        changed[key] = value
        cases.append((label, changed, False))
    for key in ('repo', 'revision', 'identity_sha256', 'shards', 'metadata'):
        changed = copy.deepcopy(manifest)
        changed['source_weight_identity'][key] = [] if key in ('shards', 'metadata') else 'wrong'
        cases.append((f'identity-{key}', changed, False))
    for group in ('shards', 'metadata'):
        for field in ('file', 'bytes', 'sha256'):
            changed = copy.deepcopy(manifest)
            changed['source_weight_identity'][group][0][field] = 1 if field == 'bytes' else 'wrong'
            cases.append((f'{group}-{field}', changed, False))
    results = []
    for label, value, accepted in cases:
        (extracted / 'manifest.json').write_text(json.dumps(value))
        result = subprocess.run([str(binary), str(extracted)], capture_output=True, text=True)
        actual = result.returncode == 0
        results.append({'case': label, 'expected_accept': accepted, 'accepted': actual,
                        'passed': actual == accepted, 'returncode': result.returncode,
                        'stdout': result.stdout, 'stderr': result.stderr})
    # Preserve the valid round-trip output for inspection.
    shutil.copy2(bundle / 'manifest.json', extracted / 'manifest.json')
    report = {'scope': 'synthetic export-manifest/ZIP/Swift-reader contract; no model conversion, source rehash, iOS store or Core ML inference',
              'compile_command': command, 'cases': results, 'passed': all(r['passed'] for r in results)}
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'cases': len(results), 'report': str(output / 'results.json')}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
