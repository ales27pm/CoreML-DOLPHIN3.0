#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

function resolveVitestEntry() {
  try {
    return require.resolve('vitest/vitest.mjs');
  } catch (error) {
    console.error('[test-runner] Unable to locate vitest executable.', error);
    process.exit(1);
  }
}

function sanitizeArgs(rawArgs) {
  const sanitized = [];
  let stripped = false;
  for (const arg of rawArgs) {
    if (arg === '--runInBand') {
      stripped = true;
      continue;
    }
    sanitized.push(arg);
  }
  if (stripped) {
    console.warn('[test-runner] Ignoring unsupported flag "--runInBand" for Vitest.');
  }
  return sanitized;
}

async function main() {
  const vitestEntry = resolveVitestEntry();
  const args = sanitizeArgs(process.argv.slice(2));

  const child = spawn(process.execPath, [vitestEntry, 'run', ...args], {
    stdio: 'inherit',
    env: process.env,
  });

  child.on('error', (error) => {
    console.error('[test-runner] Failed to launch Vitest:', error);
    process.exit(1);
  });

  child.on('exit', (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 0);
  });
}

main().catch((error) => {
  console.error('[test-runner] Unexpected error while running Vitest:', error);
  process.exit(1);
});
