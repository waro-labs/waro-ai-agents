# Database Analysis

Date: 2026-06-18

## Connection

Verified local Postgres using `api_warocol.com/.env`:

- host: `localhost`
- port: `5432`
- database: `postresWaroLabs`
- user: `saifer`
- version: PostgreSQL 16.13

Secrets were not copied into this document.

## Extensions

Installed extensions:

- `vector 0.8.1`
- `pg_cron 1.6`
- `pg_stat_statements 1.10`
- `pg_trgm 1.6`
- `unaccent 1.1`
- `uuid-ossp 1.1`
- `plpgsql`

Conclusion: pgvector is already available. We do not need a separate vector database for the first version.

## Schema Overview

Current non-system schemas:

- `public`: 206 tables, about 393 MB.
- `cron`: 2 tables, about 42 MB.
- `drizzle`: migration metadata.

Largest relevant tables:

- `tenant_ingredient_movements`: about 386k rows, 165 MB.
- `order_item_ingredients`: about 180k rows, 55 MB.
- `orders`: about 14.9k rows.
- `order_items`: about 9k rows.
- `tenant_journal_lines`, `tenant_journal_entries`.
- `ingredients`, `product`, `tenant_purchases`, `tenant_purchase_items`.

These are enough to support the first high-value agents: food cost, purchasing, inventory, and financial analysis.

## RAG Status

The `vector` extension exists and functions such as `hybrid_search` and `search_similar_documents` exist, but no active table columns of type `vector(...)` were found.

Recommendation:

- Keep current `public` schema untouched.
- Add new `rag.documents` and `rag.chunks` tables.
- Use `embedding vector(1536)` initially, matching common OpenAI small embeddings.
- Store `embedding_model` and metadata to allow future re-embedding.

## Multi-Tenant Requirements

Agent tables must carry:

- `tenant_id`
- user/profile or member identity where relevant
- `conversation_id`
- `run_id`
- `trace_id`

Sensitive fields must be redacted or summarized before storage in traces/evals unless explicitly needed for audit.

## Useful Existing Tables

Identity and permissions:

- `tenants`
- `tenant_members`
- `profile`
- `api_tokens`
- `modules`
- `tools`
- `module_tools`
- `tenant_role_module_overrides`

Operational data:

- `orders`
- `order_items`
- `order_item_ingredients`
- `product`
- `product_recipes`
- `product_base_recipes`
- `ingredients`
- `tenant_inventory`
- `tenant_ingredient_movements`
- `tenant_purchases`
- `tenant_purchase_items`
- `tenant_suppliers`

Existing eval-like/gamification tables:

- `evaluation_criteria`
- `evaluation_results`

These existing eval tables appear domain-specific and should not be reused directly for LLM evals. Use `ai.eval_*` tables instead.

## Recommended New Schemas

- `ai`: conversations, messages, runs, steps, tool calls, approvals, evals.
- `rag`: documents, chunks, embeddings.
- `audit`: sensitive AI action audit events.

See `migrations/001_ai_rag_audit_schemas.sql`.
