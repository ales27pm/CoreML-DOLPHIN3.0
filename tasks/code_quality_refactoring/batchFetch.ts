export interface BatchFetchOptions {
  readonly maxConcurrency?: number;
  readonly timeoutMs?: number;
  readonly fetchImpl?: FetchLike;
}

export interface BatchFetchFailure {
  readonly index: number;
  readonly url: string;
  readonly cause: unknown;
}

export class BatchFetchError extends AggregateError {
  public readonly failures: readonly BatchFetchFailure[];

  public constructor(failures: readonly BatchFetchFailure[]) {
    super(
      failures.map((failure) => failure.cause),
      failures
        .map(
          (failure) =>
            `Request to ${failure.url} (index ${failure.index}) failed: ${describeError(failure.cause)}`,
        )
        .join("\n"),
    );
    this.name = "BatchFetchError";
    this.failures = failures;
  }
}

interface FetchResponseLike {
  readonly ok: boolean;
  readonly status: number;
  readonly statusText?: string;
  text(): Promise<string>;
}

type FetchLike = (
  input: string | URL,
  init?: { signal?: AbortSignal },
) => Promise<FetchResponseLike>;

const DEFAULT_MAX_CONCURRENCY = 10;
const DEFAULT_TIMEOUT_MS = 10_000;

export const batchFetch = async (
  urls: readonly string[],
  options: BatchFetchOptions = {},
): Promise<string[]> => {
  if (!Array.isArray(urls)) {
    throw new TypeError("urls must be an array of request targets");
  }

  const maxConcurrency = options.maxConcurrency ?? DEFAULT_MAX_CONCURRENCY;
  if (!Number.isInteger(maxConcurrency) || maxConcurrency <= 0) {
    throw new RangeError("maxConcurrency must be a positive integer");
  }

  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new RangeError("timeoutMs must be a positive, finite number");
  }

  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  if (typeof fetchImpl !== "function") {
    throw new ReferenceError(
      "A fetch implementation must be available via options.fetchImpl or the global scope",
    );
  }

  if (urls.length === 0) {
    return [];
  }

  const results: string[] = new Array(urls.length);
  const failures: BatchFetchFailure[] = [];
  let currentIndex = 0;

  const worker = async (): Promise<void> => {
    while (true) {
      const index = currentIndex;
      if (index >= urls.length) {
        return;
      }
      currentIndex += 1;
      const url = urls[index];
      try {
        results[index] = await fetchWithTimeout(url, timeoutMs, fetchImpl);
      } catch (error) {
        failures.push({ index, url, cause: error });
      }
    }
  };

  const workerCount = Math.min(maxConcurrency, urls.length);
  await Promise.all(Array.from({ length: workerCount }, () => worker()));

  if (failures.length > 0) {
    throw new BatchFetchError(failures);
  }

  return results;
};

const fetchWithTimeout = async (
  url: string,
  timeoutMs: number,
  fetchImpl: FetchLike,
): Promise<string> => {
  const controller = new AbortController();
  const abortSignal = controller.signal;
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetchImpl(url, { signal: abortSignal });
    if (!response.ok) {
      throw new Error(
        `Request failed with status ${response.status}${response.statusText ? ` ${response.statusText}` : ""}`,
      );
    }
    return await response.text();
  } catch (error) {
    if (abortSignal.aborted) {
      throw new Error(`Request to ${url} timed out after ${timeoutMs}ms`, {
        cause: error,
      });
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
};

const describeError = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
};
