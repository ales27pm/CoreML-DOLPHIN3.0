# E-Commerce Platform Architecture

## Executive Summary

This document describes a modular e-commerce platform optimised for global
scale, omnichannel customer experiences, and strict financial compliance. The
system is composed of decoupled services stitched together through a resilient
event-driven backbone. Each component is production hardened with explicit
observability signals, disaster recovery procedures, and zero-downtime deployment
strategies.

## Core Services

### API Gateway (Envoy)
- Terminates TLS, enforces mutual TLS for service-to-service traffic, and
  validates JSON Web Tokens (JWT) issued by the User Service.
- Performs request shaping (rate limits, schema validation, request/response
  transformations) and routes traffic to downstream services.
- Emits structured access logs to the observability plane and forwards tracing
  headers to preserve distributed trace continuity.

### User Service
- Handles user lifecycle (registration, authentication, MFA, password reset).
- Integrates with an identity provider (OIDC) and hardware-backed key storage
  for secure credential handling.
- Persists data in a PostgreSQL cluster with logical replication across regions.
- Publishes user profile updates to the event bus for downstream personalization
  services.

### Catalog Service
- Manages product metadata, dynamic pricing rules, search indexing, and
  recommendation signals.
- Backed by a document database (MongoDB or DynamoDB) for flexible schema
  evolution and high read throughput.
- Streams change data capture (CDC) events into the analytics warehouse for
  near-real-time merchandising insights.

### Order Service
- Coordinates cart management, order creation, inventory reservation, and
  fulfillment orchestration.
- Persists orders in a PostgreSQL cluster with partitioning by region and
  time-based retention policies.
- Publishes order state transitions (placed, paid, fulfilled, cancelled) to the
  event bus for workflow automation and customer notifications.

### Payment Service
- Integrates with multiple payment service providers (PSPs) for redundancy.
- Normalises PSP responses into a canonical payment event schema.
- Applies fraud detection heuristics and machine learning models sourced from
  the analytics platform.
- Uses an encrypted vault (HashiCorp Vault or AWS KMS) for tokenized card data.

### Inventory Service
- Maintains stock levels per SKU and fulfillment center.
- Exposes reservation and release APIs consumed by the Order Service.
- Replicates critical tables to Redis for sub-millisecond stock checks.

### Ledger Service
- Maintains the financial ledger with double-entry accounting semantics.
- Guarantees immutability through append-only storage and periodic snapshots to
  cold storage (S3/Glacier).
- Drives reconciliation exports to the finance data lake.

### Analytics & Reporting Service
- Consumes CDC events and PSP settlements to compute business KPIs.
- Provides dashboards via Looker/Tableau and exposes REST endpoints for
  high-level summaries consumed by executives.

## Data Flow Overview

1. Client requests (web, mobile, partner API) enter through the API Gateway.
2. Authenticated traffic is routed to the Order, Catalog, or User services.
3. The Order Service writes order intents, reserves inventory, and calls the
   Payment Service.
4. The Payment Service executes PSP calls, applies fraud heuristics, and emits
   payment results back to the Order Service.
5. Successful payments trigger ledger postings and downstream fulfillment events.
6. Analytics consumes CDC streams to update dashboards and personalised content.

## Architecture Diagram

```
+-----------+      +-----------+      +-----------+      +-----------+
|   Client  | ---> |   API GW  | ---> |  Order    | ---> |  Payment  |
| (Web/App) |      | (Envoy)   |      |  Service  |      |  Service  |
+-----------+      +-----------+      +-----------+      +-----------+
       |                    |                |                   |
       |                    v                v                   v
       |              +-----------+     +-----------+      +-----------+
       |              |  Catalog  |     | Inventory |      |  Ledger   |
       |              |  Service  |     |  Service  |      |  Service  |
       |              +-----------+     +-----------+      +-----------+
       |                    |                |                   |
       v                    v                v                   v
 +-----------+       +-----------+     +-----------+      +-----------+
 |   User    |       | Analytics |     |   Event   |      | Observab. |
 |  Service  |       |  Service  |     |   Bus     |      |  Plane    |
 +-----------+       +-----------+     +-----------+      +-----------+
```

*All ASCII art rows are ≤ 80 characters wide to ensure terminal readability.*

## API Surface

| Service  | Method & Route          | Description                                                     |
| -------- | ----------------------- | ---------------------------------------------------------------- |
| Orders   | `POST /orders`          | Create an order, reserve inventory, and emit `order.placed` events. |
| Payments | `POST /payments`        | Charge customer with idempotency keys, store PSP reference IDs. |
| Products | `GET /products`         | Retrieve paginated catalog entries with optional filters.       |
| Users    | `POST /sessions`        | Initiate login; issues JWT and refresh token pair.               |
| Users    | `POST /mfa/challenge`   | Deliver time-bound MFA challenges via SMS/email/push.            |
| Inventory| `POST /inventory/check` | Validate SKU availability for a cart payload.                    |

All APIs enforce JSON schemas via the API Gateway. Idempotency keys are required
for order and payment operations to prevent duplicate processing.

## Resilience & Reliability

- **Retry Policies:** Client-facing retries use exponential backoff with jitter.
  Internal retries are bounded and instrumented with failure budgets.
- **Circuit Breakers:** Envoy limits concurrent upstream failures and returns
  cached fallbacks for catalog browsing.
- **Bulkheads:** Independent compute pools for read-heavy (catalog) and
  write-heavy (order/payment) workloads prevent resource starvation.
- **Dead-Letter Queues:** All asynchronous workflows route failed events to a
  DLQ with automated alerting and replay tooling.
- **Disaster Recovery:** Cross-region active/active deployment with RPO ≤ 5
  minutes and RTO ≤ 30 minutes via blue/green rollouts.

## Observability

- **Metrics:** Prometheus scrapes service-level metrics; SLOs defined for latency,
  error rate, and throughput.
- **Tracing:** OpenTelemetry SDK propagates traces end-to-end. Critical user flows
  (checkout, payment) have dedicated trace-based alerts.
- **Logging:** Structured JSON logs with correlation IDs shipped to ELK/Splunk.
- **Audit Trails:** Ledger and payment actions recorded in tamper-evident storage.

## Security & Compliance

- **Authentication:** OAuth 2.1 with PKCE for public clients; service-to-service
  calls use SPIFFE identities.
- **Data Protection:** PII encrypted at rest (AES-256) and in transit (TLS 1.3).
- **Compliance:** PCI DSS scope isolated to Payment and Ledger services via
  network segmentation and separate IAM boundaries.
- **Secrets Management:** All secrets stored in Vault/KMS with automated rotation.
- **Monitoring:** Continuous vulnerability scanning and runtime policy enforcement
  through Falco/OPA Gatekeeper.

## Deployment & Operations

- GitOps workflow using Argo CD ensures declarative infrastructure updates.
- CI/CD pipeline runs unit, integration, and chaos tests before promotion.
- Canary rollouts gradually shift traffic with automated rollback on SLO breach.
- Infrastructure defined via Terraform with environment-specific overlays.
- Feature flags managed by a central service (LaunchDarkly) to decouple deploys
  from releases.

## Data Management

- Polyglot persistence (PostgreSQL, document DB, Redis cache, object storage).
- CDC pipeline streams into a Snowflake warehouse for analytics.
- Data retention policies align with GDPR/CCPA, including user deletion workflows.
- Backups stored in encrypted buckets with quarterly restore drills.

## Verification Checklist

1. Render this Markdown file ensuring the ASCII diagram remains within 80 columns.
2. Review service responsibilities with platform architecture stakeholders and
   record sign-off in the change log.
3. Store the document in the internal knowledge base with semantic version tags.
