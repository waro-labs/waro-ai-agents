-- Initial WARO AI agent schemas.
-- Draft only: review before applying.

CREATE SCHEMA IF NOT EXISTS ai;
CREATE SCHEMA IF NOT EXISTS rag;
CREATE SCHEMA IF NOT EXISTS audit;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS ai.conversations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    created_by_profile_id uuid REFERENCES public.profile(id) ON DELETE SET NULL,
    title text,
    status text NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'archived', 'deleted')),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ai_conversations_tenant_updated
    ON ai.conversations (tenant_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS ai.messages (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id uuid NOT NULL REFERENCES ai.conversations(id) ON DELETE CASCADE,
    tenant_id uuid NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    role text NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content text,
    content_sanitized text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    token_count integer,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ai_messages_conversation_created
    ON ai.messages (conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ai_messages_tenant_created
    ON ai.messages (tenant_id, created_at DESC);

CREATE TABLE IF NOT EXISTS ai.runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id uuid NOT NULL REFERENCES ai.conversations(id) ON DELETE CASCADE,
    tenant_id uuid NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    input_message_id uuid REFERENCES ai.messages(id) ON DELETE SET NULL,
    output_message_id uuid REFERENCES ai.messages(id) ON DELETE SET NULL,
    trace_id text,
    agent_name text NOT NULL,
    graph_name text,
    status text NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed', 'cancelled', 'waiting_for_approval')),
    model text,
    total_input_tokens integer,
    total_output_tokens integer,
    total_cost_usd numeric(12, 6),
    error jsonb,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_ai_runs_conversation_started
    ON ai.runs (conversation_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_runs_tenant_started
    ON ai.runs (tenant_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_runs_trace_id
    ON ai.runs (trace_id);

CREATE TABLE IF NOT EXISTS ai.steps (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL REFERENCES ai.runs(id) ON DELETE CASCADE,
    tenant_id uuid NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    parent_step_id uuid REFERENCES ai.steps(id) ON DELETE SET NULL,
    span_id text,
    step_type text NOT NULL
        CHECK (step_type IN ('router', 'agent', 'llm', 'tool', 'retriever', 'eval', 'approval', 'system')),
    name text NOT NULL,
    input_json jsonb,
    output_json jsonb,
    output_summary text,
    error_json jsonb,
    latency_ms integer,
    token_usage jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ai_steps_run_created
    ON ai.steps (run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ai_steps_span_id
    ON ai.steps (span_id);

CREATE TABLE IF NOT EXISTS ai.tool_calls (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL REFERENCES ai.runs(id) ON DELETE CASCADE,
    step_id uuid REFERENCES ai.steps(id) ON DELETE SET NULL,
    tenant_id uuid NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    tool_name text NOT NULL,
    tool_version text,
    arguments_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    result_json jsonb,
    result_summary text,
    dry_run boolean NOT NULL DEFAULT false,
    approval_required boolean NOT NULL DEFAULT false,
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled', 'waiting_for_approval')),
    error_json jsonb,
    idempotency_key text,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_ai_tool_calls_run_started
    ON ai.tool_calls (run_id, started_at);
CREATE INDEX IF NOT EXISTS idx_ai_tool_calls_tenant_tool_started
    ON ai.tool_calls (tenant_id, tool_name, started_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_ai_tool_calls_idempotency
    ON ai.tool_calls (tenant_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS ai.human_approvals (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    run_id uuid NOT NULL REFERENCES ai.runs(id) ON DELETE CASCADE,
    tool_call_id uuid REFERENCES ai.tool_calls(id) ON DELETE CASCADE,
    requested_by_profile_id uuid REFERENCES public.profile(id) ON DELETE SET NULL,
    approved_by_profile_id uuid REFERENCES public.profile(id) ON DELETE SET NULL,
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'rejected', 'expired', 'cancelled')),
    request_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    decision_note text,
    created_at timestamptz NOT NULL DEFAULT now(),
    decided_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_ai_human_approvals_tenant_status
    ON ai.human_approvals (tenant_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS ai.context_summaries (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id uuid NOT NULL REFERENCES ai.conversations(id) ON DELETE CASCADE,
    tenant_id uuid NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    summary text NOT NULL,
    covered_from_message_id uuid REFERENCES ai.messages(id) ON DELETE SET NULL,
    covered_to_message_id uuid REFERENCES ai.messages(id) ON DELETE SET NULL,
    summary_type text NOT NULL DEFAULT 'compact'
        CHECK (summary_type IN ('compact', 'handoff', 'memory', 'error')),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ai_context_summaries_conversation_created
    ON ai.context_summaries (conversation_id, created_at DESC);

CREATE TABLE IF NOT EXISTS ai.eval_cases (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid REFERENCES public.tenants(id) ON DELETE CASCADE,
    name text NOT NULL,
    eval_type text NOT NULL
        CHECK (eval_type IN ('tool_usage', 'rag', 'business', 'safety', 'textual', 'regression')),
    input_json jsonb NOT NULL,
    expected_json jsonb,
    rubric jsonb NOT NULL DEFAULT '{}'::jsonb,
    tags text[] NOT NULL DEFAULT '{}'::text[],
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ai_eval_cases_type_active
    ON ai.eval_cases (eval_type, is_active);

CREATE TABLE IF NOT EXISTS ai.eval_results (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_case_id uuid REFERENCES ai.eval_cases(id) ON DELETE SET NULL,
    run_id uuid REFERENCES ai.runs(id) ON DELETE SET NULL,
    tenant_id uuid REFERENCES public.tenants(id) ON DELETE CASCADE,
    evaluator_name text NOT NULL,
    score numeric(5, 4),
    passed boolean,
    result_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ai_eval_results_run_created
    ON ai.eval_results (run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS rag.documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid REFERENCES public.tenants(id) ON DELETE CASCADE,
    source_type text NOT NULL,
    source_id text,
    title text,
    uri text,
    content_hash text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_documents_tenant_source
    ON rag.documents (tenant_id, source_type, source_id);

CREATE TABLE IF NOT EXISTS rag.chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES rag.documents(id) ON DELETE CASCADE,
    tenant_id uuid REFERENCES public.tenants(id) ON DELETE CASCADE,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    content_hash text,
    embedding_model text NOT NULL,
    embedding vector(1536),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_tenant_document
    ON rag.chunks (tenant_id, document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_hnsw
    ON rag.chunks USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;

CREATE TABLE IF NOT EXISTS audit.ai_action_events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    run_id uuid REFERENCES ai.runs(id) ON DELETE SET NULL,
    tool_call_id uuid REFERENCES ai.tool_calls(id) ON DELETE SET NULL,
    actor_profile_id uuid REFERENCES public.profile(id) ON DELETE SET NULL,
    action text NOT NULL,
    risk_level text NOT NULL DEFAULT 'low'
        CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    status text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_ai_action_events_tenant_created
    ON audit.ai_action_events (tenant_id, created_at DESC);
