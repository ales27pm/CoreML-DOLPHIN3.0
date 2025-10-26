import { describe, expect, it } from "vitest";

import {
  ProcessItemsError,
  processItems,
} from "../../tasks/code_quality_refactoring/processItems";

describe("processItems", () => {
  it("matches snapshot output for mixed item statuses", () => {
    const payload = [
      { status: "active", values: [1, 2, 3] },
      { status: "inactive", values: [4, 5] },
      { status: "active", values: [6] },
    ] as const;

    expect(processItems(payload)).toMatchSnapshot();
  });

  it("normalizes whitespace and casing when filtering active items", () => {
    const payload = [
      { status: " Active ", values: [0.5, 1.5] },
      { status: "ACTIVE", values: [2] },
      { status: "pending", values: [7] },
    ] as const;

    expect(processItems(payload)).toStrictEqual([1, 3, 4]);
  });

  it("throws a descriptive error when encountering invalid numeric entries", () => {
    const payload = [{ status: "active", values: [1, Number.NaN] }] as const;

    expect(() => processItems(payload as unknown as typeof payload)).toThrow(
      ProcessItemsError,
    );
  });
});
