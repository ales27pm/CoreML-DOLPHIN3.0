import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { enrichJsDoc } from "../../tasks/documentation/jsdoc_enricher";

describe("enrichJsDoc", () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = mkdtempSync(path.join(os.tmpdir(), "jsdoc-enricher-"));
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  it("annotates exported function declarations", () => {
    const file = path.join(tempDir, "example.ts");
    writeFileSync(
      file,
      `export function sum(a: number, b: number): number {\n  return a + b;\n}\n`,
      "utf8",
    );

    const updated = enrichJsDoc(file);
    expect(updated).toEqual([file]);

    const content = readFileSync(file, "utf8");
    expect(content).toMatch(
      /^\/\*\*[\s\S]*Auto-generated documentation for sum\./,
    );
    expect(content).toMatch(/\* @param {number} a - Parameter a\./);
    expect(content).toMatch(/\* @returns .*Return value\./);
  });

  it("skips nodes that already include documentation", () => {
    const file = path.join(tempDir, "documented.ts");
    writeFileSync(
      file,
      `/**\n * Custom docs.\n */\nexport function alreadyDocumented(): void {\n  console.log("noop");\n}\n`,
      "utf8",
    );

    const updated = enrichJsDoc(file);
    expect(updated).toHaveLength(0);
    const content = readFileSync(file, "utf8");
    expect(content).toContain("Custom docs.");
  });

  it("documents exported arrow functions", () => {
    const file = path.join(tempDir, "arrows.ts");
    writeFileSync(
      file,
      `export const format = (value: string) => value.trim();\n`,
      "utf8",
    );

    const updated = enrichJsDoc(file);
    expect(updated).toEqual([file]);
    const content = readFileSync(file, "utf8");
    expect(content).toMatch(/Auto-generated documentation for format/);
    expect(content).toMatch(/\* @param {string} value - Parameter value\./);
    expect(content).toMatch(/\* @returns .*Return value\./);
  });
});
