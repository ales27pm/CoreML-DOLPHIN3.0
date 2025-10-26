import { describe, expect, it } from "vitest";
import {
  batchFetch,
  BatchFetchError,
  type BatchFetchFailure,
  type FetchLike,
} from "../../tasks/code_quality_refactoring/batchFetch";

const createResponse = (
  body: string,
  status = 200,
): {
  ok: boolean;
  status: number;
  text: () => Promise<string>;
} => ({
  ok: status >= 200 && status < 300,
  status,
  text: async () => body,
});

describe("batchFetch", () => {
  it("resolves urls without exceeding the concurrency ceiling", async () => {
    const urls = Array.from(
      { length: 6 },
      (_, index) => `https://service/${index}`,
    );
    let active = 0;
    let observedMaximum = 0;

    const fetchImpl: FetchLike = async (input) => {
      active += 1;
      observedMaximum = Math.max(observedMaximum, active);
      try {
        await new Promise((resolve) => setTimeout(resolve, 10));
        return createResponse(String(input));
      } finally {
        active -= 1;
      }
    };

    const results = await batchFetch(urls, { maxConcurrency: 2, fetchImpl });

    expect(results).toStrictEqual(urls.map(String));
    expect(observedMaximum).toBeLessThanOrEqual(2);
  });

  it("aggregates fetch failures", async () => {
    const urls = ["https://ok", "https://fail", "https://also-ok"];

    const fetchImpl: FetchLike = async (input) => {
      if (String(input).includes("fail")) {
        throw new Error("Simulated failure");
      }
      return createResponse(`body-${input}`);
    };

    await expect(batchFetch(urls, { fetchImpl })).rejects.toMatchObject({
      name: "BatchFetchError",
      failures: [
        {
          index: 1,
          url: "https://fail",
          cause: expect.any(Error),
        },
      ] satisfies BatchFetchFailure[],
    });
  });

  it("throws when configured with an invalid concurrency", async () => {
    await expect(
      batchFetch(["https://example.com"], { maxConcurrency: 0 }),
    ).rejects.toThrowError(/maxConcurrency/);
  });

  it("fails requests that exceed the timeout budget", async () => {
    const fetchImpl: FetchLike = (_input, init) =>
      new Promise((_, reject) => {
        const signal = init?.signal;
        if (!signal) {
          reject(new Error("Missing abort signal"));
          return;
        }
        signal.addEventListener("abort", () => {
          reject(new Error("Aborted"));
        });
      });

    const promise = batchFetch(["https://slow.example"], {
      timeoutMs: 25,
      fetchImpl,
    });

    await expect(promise).rejects.toBeInstanceOf(BatchFetchError);
    await expect(promise).rejects.toThrow(/timed out/i);
  });

  it("surfaces HTTP status errors", async () => {
    const fetchImpl: FetchLike = async () => createResponse("boom", 500);

    await expect(
      batchFetch(["https://server-error"], { fetchImpl }),
    ).rejects.toThrowError(/Request failed with status 500/);
  });
});
