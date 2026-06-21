import json
from typing import Any

from app.llm.base import LLMMessage
from app.tools.sanitize import sanitize_value


def agent_router_messages(*, question: str) -> list[LLMMessage]:
    safe_payload = sanitize_value({"question": question})
    return [
        LLMMessage(
            role="system",
            content=(
                "Eres el pre-router de dominio de Kali para WARO. "
                "No respondas al usuario. Devuelve SOLO JSON valido, sin markdown. "
                "Tu unica tarea es decidir si la pregunta debe ir al workflow de ventas "
                "o al workflow de food cost. Usa sales para ventas, ordenes, ingresos, "
                "ticket promedio, clientes, productos vendidos, analisis comercial o "
                "analisis financiero basado en ventas/margenes. Preguntas como "
                "'productos que venden mucho pero tienen bajo margen' van a sales, "
                "porque cruzan ventas/cantidad con margen comercial. Usa food_cost para "
                "recetas, costos de preparacion, insumos, margen de receta, costo de "
                "comida o rentabilidad operativa por producto/receta. Si la pregunta "
                "es saludo, conversacion general o ambigua, elige sales con baja "
                "confianza. Formato exacto: "
                "{\"workflow\":\"sales|food_cost|unknown\","
                "\"confidence\":0.0,\"reason\":\"...\"}."
            ),
        ),
        LLMMessage(
            role="user",
            content=json.dumps(safe_payload, ensure_ascii=False, default=str),
        ),
    ]


def sales_planner_messages(
    *,
    question: str,
    today: str,
    timezone: str,
    tool_catalog: list[dict[str, Any]],
) -> list[LLMMessage]:
    safe_payload = sanitize_value(
        {
            "question": question,
            "today": today,
            "timezone": timezone,
            "available_tools": [
                {
                    "name": tool.get("name"),
                    "domain": tool.get("domain"),
                    "description": tool.get("description"),
                    "default_fields": tool.get("default_fields"),
                    "capabilities": tool.get("capabilities"),
                    "arguments_schema": tool.get("arguments_schema"),
                }
                for tool in tool_catalog
            ],
            "allowed_intents": ["small_talk", "sales_metrics"],
            "allowed_answer_styles": [
                "direct_metric",
                "business_analysis",
                "financial_analysis",
                "diagnostic",
            ],
            "allowed_request_kinds": [
                "direct_metric",
                "business_analysis",
                "financial_analysis",
                "product_ranking",
                "customer_ranking",
                "daily_analysis",
                "diagnostic",
            ],
            "allowed_areas": ["commercial", "finance", "menu", "customers"],
            "allowed_dimensions": ["overall", "date", "product", "customer", "category", "hour"],
            "allowed_sort_fields": {
                "product": ["quantity", "revenue", "margin", "cost"],
                "customer": ["order_count", "total_spent", "avg_ticket", "last_order_date"],
            },
            "allowed_operations": [
                "filter",
                "rank",
                "sort",
                "limit",
                "compare",
                "group",
                "aggregate",
            ],
            "allowed_group_by": [None, "date", "weekday", "hour", "product", "payment", "ticket"],
        }
    )
    return [
        LLMMessage(
            role="system",
            content=(
                "Eres el planner semantico de Kali para ventas WARO. "
                "No redactes respuesta al usuario. Devuelve SOLO JSON valido, sin markdown. "
                "Tu trabajo es resolver intencion, periodo, estilo de respuesta y llamadas a tools. "
                "Usa la fecha actual y zona horaria entregadas. Reglas criticas: "
                "'mes pasado' es el mes calendario anterior completo; 'este mes', 'del mes', "
                "'mes actual', 'presente mes' o typos cercanos como 'presnete mes' "
                "son desde el dia 1 del mes actual hasta today; 'ayer' es today menos un dia. "
                "Si el usuario pide promedio por dia, tendencia diaria, dia a dia o ultimos N dias, "
                "usa group_by='date'. Para preguntas concretas como 'ventas de ayer', "
                "answer_style debe ser direct_metric. Para analisis financiero, usa financial_analysis. "
                "Elige tools y operaciones leyendo available_tools.capabilities: entity, grain, "
                "measures, dimensions, supported_operations, default_rank, active_condition y "
                "supports_period. No dependas solo del nombre de la tool; si aparece una tool nueva "
                "con capabilities relevantes, puedes seleccionarla. "
                "Si pide productos mas vendidos, productos por cantidad o ranking de productos, "
                "request_kind='product_ranking', dimensions incluye 'product', sort_field='quantity' "
                "y tools debe incluir waro.financial.products. Si pide clientes frecuentes, mejores "
                "clientes o ranking de clientes, request_kind='customer_ranking', dimensions incluye "
                "'customer'. Usa sort_field='order_count' solo para frecuencia, ordenes o clientes "
                "frecuentes; usa sort_field='total_spent' para mejores clientes, mayor valor o "
                "clientes que mas compraron en dinero; tools debe incluir "
                "waro.customers.metrics y waro.customers.list. Si el usuario pide N elementos, "
                "devuelve limit=N. Para preguntas de ranking o comparacion, agrega operations: "
                "una lista breve de pasos analiticos como filter, rank, sort, limit, compare, "
                "group o aggregate. Ejemplo clientes activos: "
                "[{\"type\":\"filter\",\"condition\":\"order_count > 0 OR total_spent > 0\"},"
                "{\"type\":\"rank\",\"by\":[\"order_count\",\"total_spent\"],\"direction\":\"desc\"},"
                "{\"type\":\"limit\",\"value\":20}]. Tolera errores de escritura del usuario. "
                "Formato exacto: {\"intent\":\"small_talk|sales_metrics\","
                "\"date_from\":\"YYYY-MM-DD|null\",\"date_to\":\"YYYY-MM-DD|null\","
                "\"group_by\":\"date|weekday|hour|product|payment|ticket|null\","
                "\"answer_style\":\"direct_metric|business_analysis|financial_analysis|diagnostic\","
                "\"request_kind\":\"direct_metric|business_analysis|financial_analysis|product_ranking|customer_ranking|daily_analysis|diagnostic\","
                "\"area\":\"commercial|finance|menu|customers\","
                "\"objective\":\"...\","
                "\"dimensions\":[\"overall|date|product|customer|category|hour\"],"
                "\"requested_metrics\":[\"sales|average_ticket|quantity_sold|revenue|gross_profit|frequency|customer_activity\"],"
                "\"limit\":20,"
                "\"sort_field\":\"quantity|revenue|margin|cost|order_count|total_spent|avg_ticket|last_order_date|null\","
                "\"operations\":[{\"type\":\"filter|rank|sort|limit|compare|group|aggregate\","
                "\"condition\":\"...\",\"by\":[\"...\"],\"direction\":\"asc|desc\",\"value\":20}],"
                "\"tools\":[{\"name\":\"waro.sales.metrics\",\"reason\":\"...\"}],"
                "\"confidence\":0.0,\"reason\":\"...\"}."
            ),
        ),
        LLMMessage(
            role="user",
            content=json.dumps(safe_payload, ensure_ascii=False, default=str),
        ),
    ]


def food_cost_summary_messages(artifact: dict[str, Any]) -> list[LLMMessage]:
    safe_artifact = sanitize_value(
        {
            "question": artifact.get("question"),
            "period": artifact.get("period"),
            "low_margin_products": artifact.get("low_margin_products", []),
            "recommendations": artifact.get("recommendations", []),
            "tool_calls": artifact.get("tool_calls", []),
        }
    )
    return [
        LLMMessage(
            role="system",
            content=(
                "Eres un analista financiero para restaurantes WARO. "
                "Resume hallazgos de food cost en español, con tono claro, "
                "operativo y sin inventar datos que no estén en el JSON."
            ),
        ),
        LLMMessage(
            role="user",
            content=json.dumps(safe_artifact, ensure_ascii=False, default=str),
        ),
    ]


def sales_summary_messages(artifact: dict[str, Any]) -> list[LLMMessage]:
    safe_artifact = sanitize_value(
        {
            "question": artifact.get("question"),
            "answer_style": artifact.get("answer_style"),
            "period": artifact.get("period"),
            "metrics": artifact.get("metrics", {}),
            "analysis_request": artifact.get("analysis_request", {}),
            "response_contract": artifact.get("response_contract", {}),
            "analysis_execution": artifact.get("analysis_execution", {}),
            "auxiliary_context": artifact.get("auxiliary_context", {}),
            "financial_analysis": artifact.get("financial_analysis", {}),
            "highlights": artifact.get("highlights", []),
            "tool_calls": artifact.get("tool_calls", []),
        }
    )
    return [
        LLMMessage(
            role="system",
            content=(
                "Eres Kali, una analista senior de ventas para restaurantes en WARO Colombia. "
                "Responde en espanol claro, ejecutivo y util para un dueno o administrador. "
                "Usa unicamente los datos del JSON: no inventes ventas, ordenes, tickets, "
                "productos, tendencias ni comparaciones. Obedece answer_style: "
                "si es direct_metric, responde maximo en 2 frases, sin lectura comercial ni accion; "
                "si es business_analysis, entrega metricas, lectura comercial breve y una accion; "
                "obedece response_contract: si safe_to_answer es false, responde solo el error_message "
                "y no sustituyas la intencion con metricas generales; "
                "usa analysis_execution para entender filtros, ranking, limites y operaciones aplicadas; "
                "si un ranking filtro filas sin actividad, no las incluyas ni sugieras que tienen compras; "
                "explica brevemente la metrica usada cuando el usuario pida mejores, top o ranking; "
                "si es financial_analysis, usa analysis_request y financial_analysis como contrato: "
                "si analysis_quality es partial, di que es un analisis financiero parcial basado en "
                "ventas y margen bruto/productos; no afirmes rentabilidad neta, EBITDA, flujo de caja, "
                "viabilidad de largo plazo ni punto de equilibrio si aparecen en missing_metrics o "
                "unsupported_conclusions; separa hallazgos soportados, productos relevantes, limites "
                "de datos y siguiente accion. Si es diagnostic, explica "
                "el problema con datos y siguientes verificaciones. Si faltan datos, dilo de forma "
                "directa y no des una lista generica de recomendaciones. No conviertas una etiqueta "
                "Low Performance en perdida o baja rentabilidad neta si el JSON no lo soporta."
            ),
        ),
        LLMMessage(
            role="user",
            content=json.dumps(safe_artifact, ensure_ascii=False, default=str),
        ),
    ]
