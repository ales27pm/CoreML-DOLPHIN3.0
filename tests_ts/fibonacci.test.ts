import { tmpdir } from "node:os";
import { mkdtempSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildFibonacciReport,
  fibonacciParity,
  fibonacciTypeScript,
  writeFibonacciReport,
  type FibonacciReport,
} from "../tasks/multi_language_cross_integration/fibonacci";

const tempRoot = mkdtempSync(join(tmpdir(), "fib-ts-"));

describe("fibonacciTypeScript", () => {
  it("produces expected sequence for canonical inputs", () => {
    expect(fibonacciTypeScript(0)).toEqual([]);
    expect(fibonacciTypeScript(1)).toEqual([0]);
    expect(fibonacciTypeScript(7)).toEqual([0, 1, 1, 2, 3, 5, 8]);
  });

  it("memoises across calls to avoid recomputation", () => {
    const first = fibonacciTypeScript(10);
    const second = fibonacciTypeScript(12);
    expect(second.slice(0, first.length)).toEqual(first);
  });

  it("rejects non-integer requests", () => {
    expect(() => fibonacciTypeScript(3.5)).toThrow(RangeError);
  });
});

describe("fibonacciParity", () => {
  it("labels numbers correctly", () => {
    expect(fibonacciParity([0, 1, 1, 2])).toEqual([
      "even",
      "odd",
      "odd",
      "even",
    ]);
  });
});

describe("writeFibonacciReport", () => {
  it("writes the JSON payload to disk", () => {
    const target = join(tempRoot, "report.json");
    const resolved = writeFibonacciReport({
      count: 5,
      outputPath: target,
      indent: 0,
    });
    const payloadRaw: unknown = JSON.parse(readFileSync(resolved, "utf-8"));
    if (
      !payloadRaw ||
      typeof payloadRaw !== "object" ||
      !("sequence" in payloadRaw) ||
      !("parity" in payloadRaw)
    ) {
      throw new TypeError("Report JSON structure is invalid");
    }
    const payload = payloadRaw as FibonacciReport;
    expect(payload).toEqual({
      sequence: [0, 1, 1, 2, 3],
      parity: ["even", "odd", "odd", "even", "odd"],
    });
  });
});

describe("buildFibonacciReport", () => {
  it("matches python parity for the first ten numbers", () => {
    const report = buildFibonacciReport(10);
    expect(report.parity).toEqual([
      "even",
      "odd",
      "odd",
      "even",
      "odd",
      "odd",
      "even",
      "odd",
      "odd",
      "even",
    ]);
  });
});
