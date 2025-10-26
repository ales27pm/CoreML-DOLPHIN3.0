export interface Item {
  readonly status: string;
  readonly values: readonly number[];
}

export class ProcessItemsError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = "ProcessItemsError";
  }
}

const ACTIVE_STATUS = "active";

const normalizeStatus = (status: string): string => status.trim().toLowerCase();

const assertValidItem = (item: Item, index: number): void => {
  if (typeof item.status !== "string" || item.status.trim() === "") {
    throw new ProcessItemsError(
      `Item at index ${index} must include a non-empty string status.`,
    );
  }

  if (!Array.isArray(item.values)) {
    throw new ProcessItemsError(
      `Item at index ${index} must expose a values array.`,
    );
  }

  item.values.forEach((value, valueIndex) => {
    if (
      typeof value !== "number" ||
      Number.isNaN(value) ||
      !Number.isFinite(value)
    ) {
      throw new ProcessItemsError(
        `Item at index ${index} contains an invalid numeric value at position ${valueIndex}.`,
      );
    }
  });
};

const double = (value: number): number => value * 2;

export const processItems = (items: readonly Item[]): number[] => {
  items.forEach(assertValidItem);

  return items
    .filter((item) => normalizeStatus(item.status) === ACTIVE_STATUS)
    .flatMap((item) => item.values.map(double));
};
