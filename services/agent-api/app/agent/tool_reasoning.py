from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Awaitable, Callable
from uuid import UUID
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.config import Settings
from app.dependencies.internal_auth import InternalRequestContext
from app.llm.base import LLMAdapter, LLMMessage, LLMResponse
from app.llm.model_router import Complexity, model_for
from app.telemetry import mark_span_error
from app.tools import ToolCallRequest, ToolGateway
from app.tools.catalog import normalize_query
from app.tools.registry import ToolRegistry
from app.tools.sanitize import sanitize_value
from app.agent.queryspec import DATASET_RULES, QueryDatasetRule, query_dataset_rules_from_capability


MAX_TOOL_CALLS = 4
MAX_PROMPT_TOOLS = 5
TRACER = trace.get_tracer(__name__)
ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


async def run_tool_reasoning_agent(
    *,
    settings: Settings,
    llm_adapter: LLMAdapter,
    gateway: ToolGateway,
    registry: ToolRegistry,
    question: str,
    context: InternalRequestContext,
    run_id: UUID,
    conversation_messages: list[dict[str, str]] | None,
    complexity: Complexity,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    tools = await _available_tools(registry=registry, scopes=context.scopes)
    transcript: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    initial_prompt_tools = _prompt_tools(question=question, tools=tools, observations=observations)
    await _emit_progress(
        progress_callback,
        "agent_step",
        {
            "step_type": "agent",
            "name": "select_tools",
            "status": "completed",
            "available_tool_count_full": len(tools),
            "available_tool_count_prompt": len(initial_prompt_tools),
            "selected_tools": list(initial_prompt_tools.keys()),
        },
    )
    messages = _planning_messages(
        question=question,
        tools=tools,
        conversation_messages=conversation_messages or [],
        observations=observations,
    )

    for step_index in range(MAX_TOOL_CALLS):
        await _emit_progress(
            progress_callback,
            "llm_started",
            {
                "step_type": "llm",
                "name": "tool_reasoning_plan",
                "step_index": step_index + 1,
            },
        )
        decision = await _complete_json(
            settings=settings,
            llm_adapter=llm_adapter,
            messages=messages,
            complexity=complexity,
            run_id=run_id,
            tenant_id=context.tenant_id,
            step_index=step_index + 1,
            available_tool_count=len(tools),
            observation_count=len(observations),
        )
        action = str(decision.get("action") or "answer")
        if action != "call_tool":
            break
        tool_name = str(decision.get("tool_name") or "")
        arguments = decision.get("arguments") if isinstance(decision.get("arguments"), dict) else {}
        if tool_name not in tools:
            observations.append(
                {
                    "tool_name": tool_name,
                    "status": "failed",
                    "error": {"message": "unknown_or_forbidden_tool"},
                }
            )
        else:
            arguments = _clean_arguments(arguments=arguments, tool=tools[tool_name])
            await _emit_progress(
                progress_callback,
                "tool_started",
                {
                    "tool_name": tool_name,
                    "reason": "agent_tool_call",
                    "step_index": step_index + 1,
                },
            )
            try:
                response = await gateway.call(
                    request=ToolCallRequest(
                        run_id=run_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        fields=None,
                    ),
                    context=context,
                )
                observations.append(
                    {
                        "tool_name": response.tool_name,
                        "status": response.status,
                        "arguments": arguments,
                        "fields": [],
                        "result_summary": response.result_summary,
                        "result": response.result if isinstance(response.result, dict) else {},
                        "error": response.error,
                    }
                )
                await _emit_progress(
                    progress_callback,
                    "tool_finished",
                    {
                        "tool_name": response.tool_name,
                        "status": response.status,
                        "tool_call_id": str(response.tool_call_id)
                        if response.tool_call_id
                        else None,
                        "result_summary": response.result_summary,
                        "error_type": _progress_error_type(response.error),
                    },
                )
            except HTTPException as exc:
                observations.append(
                    {
                        "tool_name": tool_name,
                        "status": "failed",
                        "arguments": arguments,
                        "fields": [],
                        "result_summary": None,
                        "result": {},
                        "error": {
                            "message": str(exc.detail),
                            "status_code": exc.status_code,
                        },
                    }
                )
                await _emit_progress(
                    progress_callback,
                    "tool_finished",
                    {
                        "tool_name": tool_name,
                        "status": "failed",
                        "result_summary": None,
                        "error_type": "HTTPException",
                    },
                )
            except Exception as exc:
                observations.append(
                    {
                        "tool_name": tool_name,
                        "status": "failed",
                        "arguments": arguments,
                        "fields": [],
                        "result_summary": None,
                        "result": {},
                        "error": {"message": str(exc), "type": type(exc).__name__},
                    }
                )
                await _emit_progress(
                    progress_callback,
                    "tool_finished",
                    {
                        "tool_name": tool_name,
                        "status": "failed",
                        "result_summary": None,
                        "error_type": type(exc).__name__,
                    },
                )
        transcript.append({"step": step_index + 1, "decision": decision, "observation": observations[-1]})
        messages = _planning_messages(
            question=question,
            tools=tools,
            conversation_messages=conversation_messages or [],
            observations=observations,
        )

    await _emit_progress(
        progress_callback,
        "agent_step",
        {
            "step_type": "agent",
            "name": "compose_answer",
            "status": "started",
            "observation_count": len(observations),
        },
    )
    summary = await _final_answer(
        settings=settings,
        llm_adapter=llm_adapter,
        question=question,
        conversation_messages=conversation_messages or [],
        observations=observations,
        complexity=complexity,
        run_id=run_id,
        tenant_id=context.tenant_id,
    )
    safe_to_answer = bool(summary.strip())
    return sanitize_value(
        {
            "intent": "tool_reasoning",
            "agent_mode": True,
            "agent_engine_version": "tool-reasoning-v1",
            "question": question,
            "safe_to_answer": safe_to_answer,
            "answerability": "answerable" if safe_to_answer else "blocked",
            "summary": summary.strip(),
            "observations": observations,
            "tool_calls": observations,
            "transcript": transcript,
            "classification": {"complexity": complexity},
        }
    )


async def _available_tools(*, registry: ToolRegistry, scopes: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    snapshot = await registry.refresh()
    scope_set = set(scopes)
    result = {}
    for name, spec in snapshot.tools.items():
        if not _tool_scope_allowed(tool_name=name, scope=spec.scope, scopes=scope_set):
            continue
        schema = snapshot.schemas.get(name)
        arguments_schema = (
            schema.arguments_schema
            if schema is not None and schema.arguments_schema
            else spec.args_model.model_json_schema(by_alias=True)
        )
        if name == "waro.queries.run" and not _schema_has_named_properties(arguments_schema):
            arguments_schema = _queryspec_arguments_schema()
        result[name] = {
            "name": spec.name,
            "description": spec.description,
            "scope": spec.scope,
            "domain": spec.domain,
            "default_fields": list(spec.default_fields),
            "allowed_fields": sorted(spec.allowed_fields),
            "arguments_schema": arguments_schema,
            "capabilities": dict(spec.capabilities),
        }
        if name == "waro.queries.run":
            result[name]["queryspec_contract"] = _queryspec_contract_hint(
                query_dataset_rules_from_capability(spec) or DATASET_RULES
            )
    return result


def _clean_arguments(*, arguments: dict[str, Any], tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("name") == "waro.queries.run":
        if isinstance(arguments.get("spec"), str):
            return {"spec": arguments["spec"]}
        if isinstance(arguments.get("spec"), dict):
            return {"spec": json.dumps(arguments["spec"], ensure_ascii=False)}
        if arguments:
            return {"spec": json.dumps(arguments, ensure_ascii=False)}
    schema = tool.get("arguments_schema") if isinstance(tool.get("arguments_schema"), dict) else {}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    if not properties:
        return dict(arguments)
    return {key: value for key, value in arguments.items() if key in properties}


def _schema_has_named_properties(schema: dict[str, Any]) -> bool:
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    return bool(properties)


def _queryspec_arguments_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "spec": {
                "type": "string",
                "description": "JSON string QuerySpec seguro para /v1/queries/run.",
            }
        },
        "required": ["spec"],
    }


def _queryspec_contract_hint(rules: dict[str, QueryDatasetRule] | None = None) -> dict[str, Any]:
    rules = rules or DATASET_RULES
    datasets: dict[str, dict[str, Any]] = {}
    for name, rule in rules.items():
        datasets[name] = {
            "dimensions": sorted(rule.dimensions),
            "measures": sorted(rule.measures),
            "filters": sorted(rule.filters),
            "sortable_fields": sorted(rule.sortable_fields),
        }
    if "product_profitability" in datasets:
        datasets["product_profitability"]["example_spec"] = {
            "dataset": "product_profitability",
            "measures": [
                "quantity_sold",
                "profit_margin_pct",
                "profit_margin_real_pct",
                "revenue",
            ],
            "dimensions": ["product"],
            "filters": {"date_range": {"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}},
            "order_by": [{"field": "quantity_sold", "direction": "desc"}],
            "limit": 20,
        }
    if "customers" in datasets:
        datasets["customers"]["example_spec"] = {
            "dataset": "customers",
            "measures": ["order_count", "total_spent", "avg_ticket", "last_order_date"],
            "dimensions": ["customer"],
            "filters": {"date_range": {"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}},
            "order_by": [{"field": "order_count", "direction": "desc"}],
            "limit": 20,
        }
    return {
        "preferred_for": [
            "productos mas vendidos con margen real",
            "rentabilidad por producto",
            "clientes genericos e impacto por cliente",
            "metricas donde se necesita cruzar ventas con margen/costo",
            "inventario, bajo stock, insumos, ingredientes y vencimientos",
            "compras, proveedores y abastecimiento por QuerySpec",
        ],
        "rule": "Usa solo datasets listados en este contrato o descubiertos con waro.queries.schema; no inventes nombres de datasets.",
        "datasets": datasets,
    }


def _planning_messages(
    *,
    question: str,
    tools: dict[str, dict[str, Any]],
    conversation_messages: list[dict[str, str]],
    observations: list[dict[str, Any]],
) -> list[LLMMessage]:
    prompt_tools = _prompt_tools(question=question, tools=tools, observations=observations)
    payload = {
        "question": question,
        "current_date": _current_date_payload(),
        "conversation_messages": conversation_messages[-8:],
        "available_tools": prompt_tools,
        "observations": observations,
    }
    return [
        LLMMessage(
            role="system",
            content=(
                "Eres Kali, agente conversacional de analitica WARO para restaurantes. "
                "Piensa como un analista: usa contexto, decide si necesitas datos, llama herramientas disponibles "
                "y evita repetir respuestas anteriores. Devuelve SOLO JSON valido. "
                "Formato para llamar herramienta: {\"action\":\"call_tool\",\"tool_name\":\"...\",\"arguments\":{}}. "
                "Formato para responder sin mas herramientas: {\"action\":\"answer\"}. "
                "No inventes cifras. Usa herramientas solo si la pregunta necesita evidencia nueva. "
                "Interpreta 'ultimo año' como los ultimos 12 meses cerrando en current_date. "
                "Para productos mas vendidos con margen o rentabilidad, prefiere waro.queries.run con dataset product_profitability. "
                "Usa waro.financial.products solo si waro.queries.run no esta disponible o falla. "
                "Para clientes frecuentes o genericos usa dataset customers o las tools waro.customers.*; nunca uses datasets no listados. "
                "Si no estas seguro del dataset o campo, llama waro.queries.schema antes de waro.queries.run. "
                "Si el usuario pide clientes genericos, busca evidencia de clientes o ventas por cliente si existe. "
                "Si el usuario pide causas, compara hipotesis de precio, costo y calidad de datos usando observaciones previas."
            ),
        ),
        LLMMessage(role="user", content=json.dumps(sanitize_value(payload), ensure_ascii=False, default=str)),
    ]


async def _complete_json(
    *,
    settings: Settings,
    llm_adapter: LLMAdapter,
    messages: list[LLMMessage],
    complexity: Complexity,
    run_id: UUID,
    tenant_id: str,
    step_index: int,
    available_tool_count: int,
    observation_count: int,
) -> dict[str, Any]:
    model = model_for(settings, step="agent_step", complexity=complexity)
    with TRACER.start_as_current_span("llm.tool_reasoning.plan") as span:
        _set_llm_request_span_attributes(
            span=span,
            llm_adapter=llm_adapter,
            model=model,
            temperature=0,
            messages=messages,
            run_id=run_id,
            tenant_id=tenant_id,
            phase="plan",
        )
        span.set_attribute("waro.tool_reasoning.step_index", step_index)
        span.set_attribute("waro.tool_reasoning.available_tool_count_full", available_tool_count)
        prompt_tool_count, prompt_tools_chars = _prompt_tools_stats(messages)
        span.set_attribute("waro.tool_reasoning.available_tool_count_prompt", prompt_tool_count)
        span.set_attribute("waro.tool_reasoning.available_tools_prompt_char_count", prompt_tools_chars)
        span.set_attribute("waro.tool_reasoning.observation_count", observation_count)
        try:
            response = await llm_adapter.complete(
                messages=messages,
                temperature=0,
                model=model,
            )
        except Exception as exc:
            mark_span_error(span, exc)
            raise
        _set_llm_response_span_attributes(span=span, response=response)
        try:
            parsed = json.loads(response.content.strip())
        except json.JSONDecodeError as exc:
            span.set_attribute("waro.tool_reasoning.parse_status", "json_error")
            span.set_attribute("waro.tool_reasoning.parse_error", str(exc))
            span.set_status(Status(StatusCode.OK))
            return {"action": "answer"}
        if isinstance(parsed, dict):
            span.set_attribute("waro.tool_reasoning.parse_status", "ok")
            span.set_attribute("waro.tool_reasoning.action", str(parsed.get("action") or ""))
            span.set_status(Status(StatusCode.OK))
            return parsed
        span.set_attribute("waro.tool_reasoning.parse_status", "non_object")
        span.set_status(Status(StatusCode.OK))
        return {"action": "answer"}


async def _final_answer(
    *,
    settings: Settings,
    llm_adapter: LLMAdapter,
    question: str,
    conversation_messages: list[dict[str, str]],
    observations: list[dict[str, Any]],
    complexity: Complexity,
    run_id: UUID,
    tenant_id: str,
) -> str:
    payload = {
        "question": question,
        "current_date": _current_date_payload(),
        "conversation_messages": conversation_messages[-8:],
        "observations": observations,
    }
    messages = [
        LLMMessage(
            role="system",
            content=(
                "Responde en espanol como Kali, una analista senior conversacional. "
                "Usa solo la evidencia de observations y el contexto conversacional. "
                "No incluyas totales, ticket promedio, ganancia total o porcentajes que no aparezcan explicitamente en observations. "
                "Si hay resultados contradictorios entre herramientas, di cual fuente estas usando y por que. "
                "Cuando el usuario diga 'margen' sin aclarar, usa profit_margin_pct y llamalo margen. "
                "Usa profit_margin_real_pct solo si el usuario pide margen real, o muestralo como columna adicional llamada margen real. "
                "No repitas rankings completos si la pregunta es seguimiento. "
                "Para causas, separa precio, costo y calidad de datos. "
                "Para clientes genericos, explica impacto en lectura de recompra, frecuencia y segmentacion. "
                "Si falta evidencia, dilo y propone la siguiente consulta concreta."
            ),
        ),
        LLMMessage(role="user", content=json.dumps(sanitize_value(payload), ensure_ascii=False, default=str)),
    ]
    model = model_for(settings, step="compose", complexity=complexity)
    with TRACER.start_as_current_span("llm.tool_reasoning.final_answer") as span:
        _set_llm_request_span_attributes(
            span=span,
            llm_adapter=llm_adapter,
            model=model,
            temperature=0.2,
            messages=messages,
            run_id=run_id,
            tenant_id=tenant_id,
            phase="final_answer",
        )
        span.set_attribute("waro.tool_reasoning.observation_count", len(observations))
        span.set_attribute(
            "waro.tool_reasoning.successful_tool_count",
            sum(1 for observation in observations if observation.get("status") == "succeeded"),
        )
        try:
            response = await llm_adapter.complete(
                messages=messages,
                temperature=0.2,
                model=model,
            )
        except Exception as exc:
            mark_span_error(span, exc)
            raise
        _set_llm_response_span_attributes(span=span, response=response)
        span.set_status(Status(StatusCode.OK))
        return response.content


def _set_llm_request_span_attributes(
    *,
    span: Any,
    llm_adapter: LLMAdapter,
    model: str,
    temperature: float,
    messages: list[LLMMessage],
    run_id: UUID,
    tenant_id: str,
    phase: str,
) -> None:
    prompt_chars = sum(len(message.content) for message in messages)
    span.set_attribute("openinference.span.kind", "LLM")
    span.set_attribute("llm.provider", llm_adapter.provider)
    span.set_attribute("llm.model", model)
    span.set_attribute("llm.model_name", model)
    span.set_attribute("llm.temperature", temperature)
    span.set_attribute("llm.prompt.message_count", len(messages))
    span.set_attribute("llm.prompt.char_count", prompt_chars)
    span.set_attribute("waro.run_id", str(run_id))
    span.set_attribute("waro.tenant_id", tenant_id)
    span.set_attribute("waro.tool_reasoning.phase", phase)


def _set_llm_response_span_attributes(*, span: Any, response: LLMResponse) -> None:
    span.set_attribute("llm.model", response.model)
    span.set_attribute("llm.model_name", response.model)
    span.set_attribute("llm.response.provider", response.provider)
    span.set_attribute("llm.response.char_count", len(response.content))
    if response.input_tokens is not None:
        span.set_attribute("llm.usage.prompt_tokens", response.input_tokens)
        span.set_attribute("llm.token_count.prompt", response.input_tokens)
    if response.output_tokens is not None:
        span.set_attribute("llm.usage.completion_tokens", response.output_tokens)
        span.set_attribute("llm.token_count.completion", response.output_tokens)
    if response.total_tokens is not None:
        span.set_attribute("llm.usage.total_tokens", response.total_tokens)
        span.set_attribute("llm.token_count.total", response.total_tokens)
    if response.prompt_cost_usd is not None:
        span.set_attribute("llm.cost.prompt", response.prompt_cost_usd)
    if response.completion_cost_usd is not None:
        span.set_attribute("llm.cost.completion", response.completion_cost_usd)
    if response.estimated_cost_usd is not None:
        span.set_attribute("llm.cost.estimated_usd", response.estimated_cost_usd)
        span.set_attribute("llm.cost.total", response.estimated_cost_usd)
    span.set_attribute("llm.cost.source", response.cost_source)


def _current_date_payload() -> dict[str, str]:
    timezone = "America/Bogota"
    now = datetime.now(ZoneInfo(timezone))
    return {"date": now.date().isoformat(), "timezone": timezone}


def _prompt_tools(
    *,
    question: str,
    tools: dict[str, dict[str, Any]],
    observations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selected_names = _select_prompt_tool_names(question=question, tools=tools, observations=observations)
    return {name: _compact_tool_for_prompt(tools[name]) for name in selected_names if name in tools}


def _select_prompt_tool_names(
    *,
    question: str,
    tools: dict[str, dict[str, Any]],
    observations: list[dict[str, Any]],
) -> list[str]:
    observed = [
        str(observation.get("tool_name"))
        for observation in observations
        if observation.get("tool_name") in tools
    ]
    scored = sorted(
        ((_tool_prompt_score(question, tool), name) for name, tool in tools.items()),
        key=lambda item: (-item[0], item[1]),
    )
    names: list[str] = []
    for name in observed:
        if name not in names:
            names.append(name)
    for score, name in scored:
        if len(names) >= MAX_PROMPT_TOOLS:
            break
        if score <= 0 and names:
            continue
        if name not in names:
            names.append(name)
    if not names:
        names = [name for _, name in scored[:MAX_PROMPT_TOOLS]]
    return names[:MAX_PROMPT_TOOLS]


def _tool_prompt_score(question: str, tool: dict[str, Any]) -> int:
    normalized_question = normalize_query(question)
    haystack_parts = [
        str(tool.get("name") or ""),
        str(tool.get("description") or ""),
        str(tool.get("domain") or ""),
        " ".join(str(field) for field in tool.get("default_fields") or []),
        " ".join(str(field) for field in tool.get("allowed_fields") or []),
        json.dumps(tool.get("capabilities") or {}, ensure_ascii=False, default=str),
    ]
    normalized_haystack = normalize_query(" ".join(haystack_parts))
    score = 0
    for token in set(normalized_question.split()):
        if len(token) >= 3 and token in normalized_haystack:
            score += 1
    product_or_margin_question = any(
        term in normalized_question
        for term in (
            "producto",
            "productos",
            "margen",
            "margenes",
            "rentabilidad",
        )
    )
    customer_question = any(
        term in normalized_question
        for term in (
            "generico",
            "genericos",
            "cliente",
            "clientes",
            "frecuente",
            "frecuentes",
            "recompra",
        )
    )
    procurement_question = any(
        term in normalized_question
        for term in (
            "inventario",
            "stock",
            "insumo",
            "insumos",
            "ingrediente",
            "ingredientes",
            "abastecimiento",
            "compra",
            "compras",
            "proveedor",
            "proveedores",
            "vencimiento",
            "vencimientos",
            "receta",
            "recetas",
        )
    )
    if tool.get("name") == "waro.queries.run" and (product_or_margin_question or customer_question):
        score += 8
    if tool.get("name") == "waro.queries.run" and procurement_question:
        score += 8
    if tool.get("name") == "waro.queries.schema" and customer_question:
        score += 5
    if tool.get("name") == "waro.queries.schema" and procurement_question:
        score += 5
    if tool.get("name") == "waro.financial.products" and product_or_margin_question:
        score += 7
    if tool.get("name") in {"waro.customers.list", "waro.customers.metrics"} and customer_question:
        score += 7
    if tool.get("domain") == "sales" and any(
        term in normalized_question for term in ("venta", "ventas", "ingreso", "ingresos", "ticket")
    ):
        score += 4
    if tool.get("domain") == "analytics" and any(
        term in normalized_question for term in ("food", "costo", "costos", "churn", "cohorte", "rfm")
    ):
        score += 4
    if tool.get("domain") in {"inventory", "purchases", "suppliers", "procurement"} and procurement_question:
        score += 7
    return score


def _tool_scope_allowed(*, tool_name: str, scope: str, scopes: set[str]) -> bool:
    if scope in scopes:
        return True
    return tool_name == "waro.queries.schema" and scope == "read" and "dataset_scope" in scopes


def _compact_tool_for_prompt(tool: dict[str, Any]) -> dict[str, Any]:
    schema = tool.get("arguments_schema") if isinstance(tool.get("arguments_schema"), dict) else {}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    optional_args = [key for key in properties.keys() if key not in required]
    payload: dict[str, Any] = {
        "name": tool.get("name"),
        "description": tool.get("description"),
        "scope": tool.get("scope"),
        "domain": tool.get("domain"),
        "arguments": {
            "required": list(required),
            "optional": optional_args[:12],
        },
        "default_fields": list(tool.get("default_fields") or [])[:12],
        "allowed_fields": list(tool.get("allowed_fields") or [])[:16],
        "capabilities": _compact_capabilities(tool.get("capabilities")),
    }
    if tool.get("queryspec_contract"):
        payload["queryspec_contract"] = tool["queryspec_contract"]
    return payload


def _compact_capabilities(value: Any, *, depth: int = 0) -> Any:
    if depth >= 2:
        if isinstance(value, dict):
            return list(value.keys())[:12]
        if isinstance(value, list):
            return value[:12]
        return value
    if isinstance(value, dict):
        return {
            str(key): _compact_capabilities(item, depth=depth + 1)
            for key, item in list(value.items())[:12]
        }
    if isinstance(value, list):
        return [_compact_capabilities(item, depth=depth + 1) for item in value[:12]]
    return value


def _prompt_tools_stats(messages: list[LLMMessage]) -> tuple[int, int]:
    for message in reversed(messages):
        if message.role != "user":
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            return 0, 0
        tools = payload.get("available_tools") if isinstance(payload, dict) else None
        if not isinstance(tools, dict):
            return 0, 0
        return len(tools), len(json.dumps(tools, ensure_ascii=False, default=str))
    return 0, 0


async def _emit_progress(
    callback: ProgressCallback | None,
    event: str,
    data: dict[str, Any],
) -> None:
    if callback is None:
        return
    await callback(event, sanitize_value(data))


def _progress_error_type(error: Any) -> str | None:
    if not error:
        return None
    if isinstance(error, dict):
        return str(error.get("type") or error.get("error") or error.get("code") or "tool_error")
    return type(error).__name__
