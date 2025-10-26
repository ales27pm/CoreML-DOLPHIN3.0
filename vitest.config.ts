import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["tests_ts/**/*.test.ts", "tests_ts/**/*.test.tsx"],
  },
});
