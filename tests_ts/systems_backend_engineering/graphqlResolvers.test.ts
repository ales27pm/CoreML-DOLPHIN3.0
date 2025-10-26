import { describe, expect, it, vi } from "vitest";

import {
  createResolverContext,
  resolvers,
  ProductNotFoundError,
  type BatchMetrics,
  type DatabaseClient,
  type OrderItemFindManyArgs,
  type OrderRepository,
  type OrderItemRepository,
  type ProductRepository,
  type ProductFindManyArgs,
  type ProductLoaderMetrics,
} from "../../tasks/systems_backend_engineering/graphqlResolvers";

describe("GraphQL resolvers", () => {
  it("batches product lookups and caches results", async () => {
    const batches: BatchMetrics[] = [];
    const cacheHits: number[] = [];

    const metrics: ProductLoaderMetrics = {
      recordBatch: (details) => {
        batches.push(details);
      },
      recordCacheHit: (key) => {
        cacheHits.push(key);
      },
    };

    const productRepo: ProductRepository = {
      findMany: vi.fn(async (args: ProductFindManyArgs) => {
        const ids = args.where.id.in;
        return ids.map((id) => ({
          id,
          sku: `SKU-${id}`,
          name: `Product ${id}`,
          priceCents: id * 100,
        }));
      }),
    };

    const orderRepo: OrderRepository = {
      findMany: vi.fn(async () => [
        { id: 1, userId: 99, createdAt: new Date("2024-01-01T00:00:00Z") },
      ]),
    };

    const orderItems = [
      { id: 1, orderId: 1, productId: 10, quantity: 2 },
      { id: 2, orderId: 1, productId: 20, quantity: 1 },
      { id: 3, orderId: 1, productId: 10, quantity: 1 },
    ] as const;

    const itemRepo: OrderItemRepository = {
      findMany: vi.fn(async (args: OrderItemFindManyArgs) =>
        orderItems.filter((item) => item.orderId === args.where.orderId),
      ),
    };

    const db: DatabaseClient = {
      product: productRepo,
      order: orderRepo,
      item: itemRepo,
    };

    const context = createResolverContext(db, { metrics });

    const orders = await resolvers.Query.orders(undefined, {}, context);
    expect(orders).toHaveLength(1);

    const items = await resolvers.Order.items(orders[0], {}, context);
    expect(items).toHaveLength(orderItems.length);

    const [first, second, third] = await Promise.all([
      resolvers.OrderItem.product(items[0], {}, context),
      resolvers.OrderItem.product(items[1], {}, context),
      resolvers.OrderItem.product(items[2], {}, context),
    ]);

    expect(first.id).toBe(10);
    expect(second.id).toBe(20);
    expect(third.id).toBe(10);

    expect(productRepo.findMany).toHaveBeenCalledTimes(1);
    expect(productRepo.findMany).toHaveBeenCalledWith({
      where: { id: { in: [10, 20] } },
    });

    expect(batches).toHaveLength(1);
    expect(batches[0].keys).toEqual([10, 20]);
    expect(batches[0].durationMs).toBeGreaterThanOrEqual(0);

    expect(cacheHits).toContain(10);
    expect(cacheHits.length).toBeGreaterThan(0);

    expect(itemRepo.findMany).toHaveBeenCalledWith({ where: { orderId: 1 } });
  });

  it("throws ProductNotFoundError when database omits a product", async () => {
    const productRepo: ProductRepository = {
      findMany: vi.fn(async (_args: ProductFindManyArgs) => []),
    };

    const orderRepo: OrderRepository = {
      findMany: vi.fn(async () => [
        { id: 1, userId: 1, createdAt: new Date("2024-01-01T00:00:00Z") },
      ]),
    };

    const itemRepo: OrderItemRepository = {
      findMany: vi.fn(async (_args: OrderItemFindManyArgs) => [
        { id: 1, orderId: 1, productId: 99, quantity: 1 },
      ]),
    };

    const db: DatabaseClient = {
      product: productRepo,
      order: orderRepo,
      item: itemRepo,
    };

    const context = createResolverContext(db);

    const orders = await resolvers.Query.orders(undefined, {}, context);
    const items = await resolvers.Order.items(orders[0], {}, context);

    await expect(
      resolvers.OrderItem.product(items[0], {}, context),
    ).rejects.toBeInstanceOf(ProductNotFoundError);

    expect(productRepo.findMany).toHaveBeenCalledTimes(1);
  });
});
