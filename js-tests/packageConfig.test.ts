import { readFileSync } from "node:fs";
import path from "node:path";

type ScriptMap = Record<string, string>;

interface PackageJson {
  scripts?: ScriptMap;
}

function loadPackageJson(): PackageJson {
  const packagePath = path.resolve(__dirname, "..", "package.json");
  const rawContents = readFileSync(packagePath, "utf8");
  try {
    return JSON.parse(rawContents) as PackageJson;
  } catch (error) {
    throw new Error(
      `Unable to parse package.json for test verification: ${(error as Error).message}`,
    );
  }
}

describe("project npm scripts", () => {
  let scripts: ScriptMap;

  beforeAll(() => {
    const packageJson = loadPackageJson();
    expect(packageJson.scripts).toBeDefined();
    scripts = packageJson.scripts ?? {};
  });

  it("exposes a lint command for static analysis", () => {
    expect(scripts.lint).toBeDefined();
    expect(scripts.lint).toContain("eslint");
  });

  it("provides a jest-backed test runner", () => {
    expect(scripts.test).toBeDefined();
    expect(scripts.test).toMatch(/jest/);
  });
});
