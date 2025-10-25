import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

export type ParityLabel = "even" | "odd";

export interface FibonacciReport {
  sequence: number[];
  parity: ParityLabel[];
}

const memoisedSequence: number[] = [0, 1];

/**
 * Generate the first `n` Fibonacci numbers.
 *
 * The implementation relies on an iterative approach backed by a module-level
 * cache so subsequent invocations are O(1) for previously requested lengths.
 * @param n Total numbers to generate.
 * @returns An immutable copy of the sequence.
 * @throws RangeError When `n` is negative or not an integer.
 */
export function fibonacciTypeScript(n: number): number[] {
  if (!Number.isInteger(n)) {
    throw new RangeError("n must be an integer");
  }
  if (n < 0) {
    throw new RangeError("n must be non-negative");
  }
  if (n === 0) {
    return [];
  }
  while (memoisedSequence.length < n) {
    const nextValue =
      memoisedSequence[memoisedSequence.length - 1] +
      memoisedSequence[memoisedSequence.length - 2];
    memoisedSequence.push(nextValue);
  }
  return memoisedSequence.slice(0, n);
}

export function fibonacciParity(sequence: Iterable<number>): ParityLabel[] {
  const labels: ParityLabel[] = [];
  for (const value of sequence) {
    labels.push(value % 2 === 0 ? "even" : "odd");
  }
  return labels;
}

export function buildFibonacciReport(count: number): FibonacciReport {
  const sequence = fibonacciTypeScript(count);
  return {
    sequence: [...sequence],
    parity: fibonacciParity(sequence),
  };
}

export interface WriteReportOptions {
  count: number;
  outputPath: string;
  indent?: number;
}

export function writeFibonacciReport(options: WriteReportOptions): string {
  const { count, outputPath, indent = 2 } = options;
  const report = buildFibonacciReport(count);
  const resolvedPath = resolve(outputPath);
  const directory = dirname(resolvedPath);
  if (directory) {
    mkdirSync(directory, { recursive: true });
  }
  writeFileSync(
    resolvedPath,
    `${JSON.stringify(report, null, indent)}\n`,
    "utf-8",
  );
  return resolvedPath;
}

function parseArguments(argv: string[]): WriteReportOptions {
  let count = 25;
  let outputPath = "";
  let indent = 2;
  for (let i = 0; i < argv.length; i += 1) {
    const current = argv[i];
    if (current === "--count") {
      const value = Number(argv[i + 1]);
      if (Number.isNaN(value)) {
        throw new Error("--count must be followed by an integer");
      }
      count = value;
      i += 1;
    } else if (current === "--output") {
      outputPath = argv[i + 1] ?? "";
      if (!outputPath) {
        throw new Error("--output requires a file path");
      }
      i += 1;
    } else if (current === "--indent") {
      const value = Number(argv[i + 1]);
      if (!Number.isInteger(value) || value < 0) {
        throw new Error("--indent must be a non-negative integer");
      }
      indent = value;
      i += 1;
    }
  }
  if (!outputPath) {
    throw new Error("--output is required");
  }
  if (!Number.isInteger(count) || count < 0) {
    throw new Error("--count must be a non-negative integer");
  }
  return { count, outputPath, indent };
}

if (require.main === module) {
  try {
    const options = parseArguments(process.argv.slice(2));
    const targetPath = writeFibonacciReport(options);
    console.log(`Fibonacci report written to ${targetPath}`);
  } catch (error) {
    console.error((error as Error).message);
    process.exitCode = 1;
  }
}
