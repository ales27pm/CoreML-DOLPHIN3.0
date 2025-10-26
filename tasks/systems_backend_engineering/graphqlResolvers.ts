import DataLoader from "dataloader";
import { performance } from "node:perf_hooks";

export interface Product {
  readonly id: number;
  readonly sku: string;
  readonly name: string;
  readonly priceCents: number;
}

export interface OrderItem {
  readonly id: number;
  readonly orderId: number;
  readonly productId: number;
  readonly quantity: number;
}

export interface Order {
  readonly id: number;
  readonly userId: number;
  readonly createdAt: Date;
}

export interface ProductFindManyArgs {
  readonly where: {
    readonly id: {
      readonly in: readonly number[];
    };
  };
}

export interface OrderItemFindManyArgs {
  readonly where: {
    readonly orderId: number;
  };
}

export interface ProductRepository {
  findMany(args: ProductFindManyArgs): Promise<readonly Product[]>;
}

export interface OrderRepository {
  findMany(): Promise<readonly Order[]>;
}

export interface OrderItemRepository {
  findMany(args: OrderItemFindManyArgs): Promise<readonly OrderItem[]>;
}

export interface DatabaseClient {
  readonly product: ProductRepository;
  readonly order: OrderRepository;
  readonly item: OrderItemRepository;
}

export interface BatchMetrics {
  readonly keys: readonly number[];
  readonly durationMs: number;
}

export interface ProductLoaderMetrics {
  recordBatch?(metrics: BatchMetrics): void;
  recordCacheHit?(key: number): void;
}

class InstrumentedCache<K extends number, V> implements Map<K, V> {
  private readonly delegate = new Map<K, V>();

  public constructor(private readonly metrics?: ProductLoaderMetrics) {}

  public get size(): number {
    return this.delegate.size;
  }

  public clear(): void {
    this.delegate.clear();
  }

  public delete(key: K): boolean {
    return this.delegate.delete(key);
  }

  public entries(): IterableIterator<[K, V]> {
    return this.delegate.entries();
  }

  public forEach(
    callbackfn: (value: V, key: K, map: Map<K, V>) => void,
    thisArg?: unknown,
  ): void {
    if (thisArg === undefined) {
      this.delegate.forEach((value, key) => {
        callbackfn(value, key, this);
      });
      return;
    }

    this.delegate.forEach((value, key) => {
      callbackfn.call(thisArg as object, value, key, this);
    });
  }

  public get(key: K): V | undefined {
    const value = this.delegate.get(key);
    if (value !== undefined) {
      this.metrics?.recordCacheHit?.(key);
    }
    return value;
  }

  public has(key: K): boolean {
    return this.delegate.has(key);
  }

  public keys(): IterableIterator<K> {
    return this.delegate.keys();
  }

  public set(key: K, value: V): this {
    this.delegate.set(key, value);
    return this;
  }

  public values(): IterableIterator<V> {
    return this.delegate.values();
  }

  public [Symbol.iterator](): IterableIterator<[K, V]> {
    return this.delegate[Symbol.iterator]();
  }

  public get [Symbol.toStringTag](): string {
    return this.delegate[Symbol.toStringTag];
  }
}

export class ProductNotFoundError extends Error {
  public constructor(public readonly productId: number) {
    super(`No product found for id ${productId}`);
    this.name = "ProductNotFoundError";
  }
}

const uniqueNumbers = (values: readonly number[]): number[] => {
  const seen = new Set<number>();
  const result: number[] = [];
  values.forEach((value) => {
    if (!seen.has(value)) {
      seen.add(value);
      result.push(value);
    }
  });
  return result;
};

const createProductLoader = (
  db: DatabaseClient,
  metrics?: ProductLoaderMetrics,
): DataLoader<number, Product> => {
  const cacheMap = new InstrumentedCache<number, Promise<Product>>(metrics);
  return new DataLoader<number, Product>(
    async (keys) => {
      const uniqueKeys = uniqueNumbers(keys);
      if (uniqueKeys.length === 0) {
        return [];
      }

      const start = performance.now();
      const products = await db.product.findMany({
        where: { id: { in: uniqueKeys } },
      });
      const durationMs = performance.now() - start;
      metrics?.recordBatch?.({ keys: uniqueKeys, durationMs });

      const productMap = new Map<number, Product>(
        products.map((product) => [product.id, product]),
      );

      return keys.map((key) => {
        const product = productMap.get(key);
        if (!product) {
          throw new ProductNotFoundError(key);
        }
        return product;
      });
    },
    { cacheMap },
  );
};

type EmptyArgs = Record<string, never>;

export interface Loaders {
  readonly productById: DataLoader<number, Product>;
}

export interface GraphQLContext {
  readonly db: DatabaseClient;
  readonly loaders: Loaders;
}

type Resolver<Parent, Args, Result> = (
  parent: Parent,
  args: Args,
  context: GraphQLContext,
) => Promise<Result> | Result;

export interface Resolvers {
  readonly Query: {
    readonly orders: Resolver<undefined, EmptyArgs, readonly Order[]>;
  };
  readonly Order: {
    readonly items: Resolver<Order, EmptyArgs, readonly OrderItem[]>;
  };
  readonly OrderItem: {
    readonly product: Resolver<OrderItem, EmptyArgs, Product>;
  };
}

export interface ResolverContextOptions {
  readonly metrics?: ProductLoaderMetrics;
}

export const createResolverContext = (
  db: DatabaseClient,
  options: ResolverContextOptions = {},
): GraphQLContext => ({
  db,
  loaders: {
    productById: createProductLoader(db, options.metrics),
  },
});

export const resolvers: Resolvers = {
  Query: {
    orders: async (_parent, _args, context) => context.db.order.findMany(),
  },
  Order: {
    items: async (parent, _args, context) =>
      context.db.item.findMany({ where: { orderId: parent.id } }),
  },
  OrderItem: {
    product: (parent, _args, context) =>
      context.loaders.productById.load(parent.productId),
  },
};
