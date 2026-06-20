from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    flag_args: ClassVar[frozenset[str]] = frozenset({"all"})

    def cli_args(self) -> list[str]:
        args: list[str] = []
        data = self.model_dump(by_alias=True, exclude_none=True)
        for key, value in data.items():
            if value is False:
                continue
            args.append(f"--{key}")
            if isinstance(value, bool):
                if key not in self.flag_args:
                    args.append(str(value).lower())
            else:
                args.append(str(value))
        return args


DateString = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]


class FoodCostArgs(ToolArgs):
    date_from: DateString | None = Field(default=None, alias="date-from")
    date_to: DateString | None = Field(default=None, alias="date-to")
    compare_to: str | None = Field(default=None, alias="compare-to", max_length=80)


class MenuProductsArgs(ToolArgs):
    flag_args: ClassVar[frozenset[str]] = ToolArgs.flag_args | frozenset(
        {"include-ingredients", "include-recipe-bases", "include-modifiers"}
    )

    limit: int = Field(default=50, ge=1, le=250)
    offset: int = Field(default=0, ge=0)
    all: bool = False
    category_id: UUID | None = Field(default=None, alias="category-id")
    is_available: bool | None = Field(default=None, alias="is-available")
    include_ingredients: bool = Field(default=True, alias="include-ingredients")
    include_recipe_bases: bool = Field(default=True, alias="include-recipe-bases")
    include_modifiers: bool = Field(default=True, alias="include-modifiers")


class MenuRecipesArgs(ToolArgs):
    limit: int = Field(default=50, ge=1, le=250)
    offset: int = Field(default=0, ge=0)
    all: bool = False
    is_active: bool | None = Field(default=None, alias="is-active")


class MenuModifiersArgs(ToolArgs):
    limit: int = Field(default=50, ge=1, le=250)
    offset: int = Field(default=0, ge=0)
    all: bool = False


class AnalyticsMenuArgs(ToolArgs):
    date_from: DateString | None = Field(default=None, alias="date-from")
    date_to: DateString | None = Field(default=None, alias="date-to")
    limit: int = Field(default=20, ge=1, le=100)


class AnalyticsAlertsArgs(ToolArgs):
    limit: int = Field(default=20, ge=1, le=100)


class AnalyticsDataQualityArgs(ToolArgs):
    pass


class CustomersListArgs(ToolArgs):
    limit: int = Field(default=50, ge=1, le=250)
    offset: int = Field(default=0, ge=0)
    all: bool = False
    search: str | None = Field(default=None, max_length=120)
    date_from: DateString | None = Field(default=None, alias="date-from")
    date_to: DateString | None = Field(default=None, alias="date-to")
    timezone: str = Field(default="America/Bogota", max_length=64)
    sort_field: Literal[
        "total_spent",
        "order_count",
        "last_order_date",
        "avg_ticket",
        "waros_balance",
    ] = Field(default="total_spent", alias="sort-field")
    sort_direction: Literal["asc", "desc"] = Field(default="desc", alias="sort-direction")


class CustomersMetricsArgs(ToolArgs):
    date_from: DateString | None = Field(default=None, alias="date-from")
    date_to: DateString | None = Field(default=None, alias="date-to")
    group_by: Literal["date", "weekday", "month"] | None = Field(
        default=None,
        alias="group-by",
    )
    timezone: str = Field(default="America/Bogota", max_length=64)


class SalesListArgs(ToolArgs):
    limit: int = Field(default=50, ge=1, le=250)
    offset: int = Field(default=0, ge=0)
    all: bool = False
    payment_method: Literal["cash", "card", "digital"] | None = Field(
        default=None,
        alias="payment-method",
    )
    status: Literal["completed", "cancelled", "pending"] | None = None
    date_from: DateString | None = Field(default=None, alias="date-from")
    date_to: DateString | None = Field(default=None, alias="date-to")
    timezone: str = Field(default="America/Bogota", max_length=64)
    sort_field: Literal[
        "order_date",
        "order_number",
        "total_amount",
        "customer_name",
        "payment_method",
    ] = Field(default="order_date", alias="sort-field")
    sort_direction: Literal["asc", "desc"] = Field(default="desc", alias="sort-direction")


class SalesMetricsArgs(ToolArgs):
    date_from: DateString | None = Field(default=None, alias="date-from")
    date_to: DateString | None = Field(default=None, alias="date-to")
    group_by: Literal["date", "weekday", "hour", "product", "payment", "ticket"] | None = Field(
        default=None,
        alias="group-by",
    )
    timezone: str = Field(default="America/Bogota", max_length=64)
    limit: int = Field(default=20, ge=1, le=100)
    sort_by: Literal["quantity", "revenue"] = Field(default="quantity", alias="sort-by")
    ranges: str | None = Field(default=None, max_length=120)
    compare_to: str | None = Field(default=None, alias="compare-to", max_length=80)


class SalesDetailArgs(ToolArgs):
    order_id: UUID = Field(alias="order-id")


class FinancialProductsArgs(ToolArgs):
    period: int = Field(default=365, ge=1, le=730)
    sort_by: Literal["margin", "revenue", "cost", "quantity"] = Field(
        default="margin",
        alias="sort-by",
    )
    min_margin: int | None = Field(default=None, alias="min-margin")
    category: str | None = Field(default=None, max_length=120)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    command: tuple[str, str]
    scope: str
    args_model: type[ToolArgs]
    default_fields: tuple[str, ...]
    allowed_fields: frozenset[str]
    domain: Literal["sales", "food_cost", "menu", "financial", "analytics", "customers"]
    description: str
    tags: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()


TOOL_SPECS: Mapping[str, ToolSpec] = {
    "waro.analytics.food_cost": ToolSpec(
        name="waro.analytics.food_cost",
        command=("analytics", "food-cost"),
        scope="analytics:read",
        args_model=FoodCostArgs,
        default_fields=("product_id", "product_name", "food_cost_pct", "margin_pct"),
        allowed_fields=frozenset(
            {"product_id", "product_name", "food_cost_pct", "margin_pct", "revenue", "cost"}
        ),
        domain="food_cost",
        description="Analyze product food cost, margin, revenue, and cost by period.",
        tags=("food cost", "margin", "profitability", "products"),
        examples=("productos con peor margen", "food cost del mes"),
    ),
    "waro.menu.products": ToolSpec(
        name="waro.menu.products",
        command=("menu", "products"),
        scope="menu:read",
        args_model=MenuProductsArgs,
        default_fields=("id", "name", "price", "isAvailable"),
        allowed_fields=frozenset(
            {
                "id",
                "name",
                "price",
                "isAvailable",
                "category",
                "calculatedCost",
                "perceivedCost",
                "preparationTime",
            }
        ),
        domain="menu",
        description="List menu products and product attributes, including price and availability.",
        tags=("menu", "products", "availability", "price"),
        examples=("productos disponibles", "lista de productos del menu"),
    ),
    "waro.menu.recipes": ToolSpec(
        name="waro.menu.recipes",
        command=("menu", "recipes"),
        scope="menu:read",
        args_model=MenuRecipesArgs,
        default_fields=("id", "name", "isActive", "ingredients"),
        allowed_fields=frozenset(
            {"id", "name", "isActive", "ingredients", "description", "createdAt", "updatedAt"}
        ),
        domain="menu",
        description="List menu recipes and recipe cost/activity metadata.",
        tags=("menu", "recipes", "ingredients", "cost"),
        examples=("recetas activas", "costos de recetas"),
    ),
    "waro.menu.modifiers": ToolSpec(
        name="waro.menu.modifiers",
        command=("menu", "modifiers"),
        scope="menu:read",
        args_model=MenuModifiersArgs,
        default_fields=("data", "meta", "success"),
        allowed_fields=frozenset({"data", "meta", "success"}),
        domain="menu",
        description="List menu modifiers and add-on options.",
        tags=("menu", "modifiers", "addons", "options"),
        examples=("modificadores del menu", "opciones adicionales"),
    ),
    "waro.analytics.menu": ToolSpec(
        name="waro.analytics.menu",
        command=("analytics", "menu"),
        scope="analytics:read",
        args_model=AnalyticsMenuArgs,
        default_fields=("data", "meta", "success"),
        allowed_fields=frozenset({"data", "meta", "success"}),
        domain="analytics",
        description="Analyze menu portfolio performance and product classifications.",
        tags=("analytics", "menu", "portfolio", "products", "performance"),
        examples=("analisis del menu", "productos estrella y bajo rendimiento"),
    ),
    "waro.analytics.alerts": ToolSpec(
        name="waro.analytics.alerts",
        command=("analytics", "alerts"),
        scope="analytics:read",
        args_model=AnalyticsAlertsArgs,
        default_fields=("data", "meta", "success"),
        allowed_fields=frozenset({"data", "meta", "success"}),
        domain="analytics",
        description="Fetch analytics alerts and operational warnings.",
        tags=("analytics", "alerts", "warnings", "operations"),
        examples=("alertas del negocio", "alertas de inventario"),
    ),
    "waro.analytics.data_quality": ToolSpec(
        name="waro.analytics.data_quality",
        command=("analytics", "data-quality"),
        scope="analytics:read",
        args_model=AnalyticsDataQualityArgs,
        default_fields=("data", "meta", "success"),
        allowed_fields=frozenset({"data", "meta", "success"}),
        domain="analytics",
        description="Check data quality signals and anomalies for analytics inputs.",
        tags=("analytics", "data quality", "anomalies", "validation"),
        examples=("calidad de datos", "datos raros o anomalos"),
    ),
    "waro.customers.list": ToolSpec(
        name="waro.customers.list",
        command=("customers", "list"),
        scope="customers:read",
        args_model=CustomersListArgs,
        default_fields=(
            "customer_id",
            "name",
            "phone",
            "order_count",
            "total_spent",
            "avg_ticket",
            "last_order_date",
            "waros_balance",
        ),
        allowed_fields=frozenset(
            {
                "customer_id",
                "name",
                "phone",
                "order_count",
                "total_spent",
                "avg_ticket",
                "last_order_date",
                "waros_balance",
            }
        ),
        domain="customers",
        description="List customers ranked by spend, orders, recency, or average ticket.",
        tags=("customers", "clients", "spend", "orders", "recency"),
        examples=("mejores clientes", "clientes por gasto"),
    ),
    "waro.customers.metrics": ToolSpec(
        name="waro.customers.metrics",
        command=("customers", "metrics"),
        scope="customers:read",
        args_model=CustomersMetricsArgs,
        default_fields=("summary", "top_customers"),
        allowed_fields=frozenset({"summary", "top_customers"}),
        domain="customers",
        description="Compute customer metrics such as activity, retention, and grouped trends.",
        tags=("customers", "metrics", "retention", "frequency", "activity"),
        examples=("metricas de clientes", "retencion de clientes"),
    ),
    "waro.sales.list": ToolSpec(
        name="waro.sales.list",
        command=("sales", "list"),
        scope="orders:read",
        args_model=SalesListArgs,
        default_fields=("id", "status", "totalAmount", "orderDate"),
        allowed_fields=frozenset(
            {
                "id",
                "status",
                "totalAmount",
                "orderDate",
                "paymentMethod",
                "customer",
                "orderNumber",
                "itemsCount",
                "items",
            }
        ),
        domain="sales",
        description="List sales orders with status, payment method, customer, date, and total.",
        tags=("sales", "orders", "tickets", "payment"),
        examples=("ordenes de ayer", "ventas canceladas"),
    ),
    "waro.sales.metrics": ToolSpec(
        name="waro.sales.metrics",
        command=("sales", "metrics"),
        scope="orders:read",
        args_model=SalesMetricsArgs,
        default_fields=("data", "meta", "success"),
        allowed_fields=frozenset(
            {
                "data",
                "meta",
                "success",
                "totalSales",
                "totalOrders",
                "orderCount",
                "avgTicket",
                "series",
                "products",
                "payment_methods",
            }
        ),
        domain="sales",
        description="Compute sales metrics such as total sales, order count, average ticket, and grouped series.",
        tags=("sales", "metrics", "revenue", "orders", "ticket", "series"),
        examples=("ventas de ayer", "ventas por hora", "ventas del mes"),
    ),
    "waro.sales.detail": ToolSpec(
        name="waro.sales.detail",
        command=("sales", "detail"),
        scope="orders:read",
        args_model=SalesDetailArgs,
        default_fields=("id", "status", "total", "items"),
        allowed_fields=frozenset({"id", "status", "total", "items", "order_date"}),
        domain="sales",
        description="Fetch details for one sales order by order id.",
        tags=("sales", "order detail", "items"),
        examples=("detalle de una orden"),
    ),
    "waro.financial.products": ToolSpec(
        name="waro.financial.products",
        command=("financial", "products"),
        scope="financial:read",
        args_model=FinancialProductsArgs,
        default_fields=("products", "metrics", "insights"),
        allowed_fields=frozenset(
            {"products", "metrics", "insights", "filters", "categories"}
        ),
        domain="financial",
        description="Rank products by margin, revenue, cost, or quantity over a period.",
        tags=("financial", "products", "margin", "revenue", "cost", "quantity"),
        examples=("productos con peor margen", "productos por ingresos"),
    ),
}


def get_tool_spec(tool_name: str) -> ToolSpec | None:
    return TOOL_SPECS.get(tool_name)


def coerce_args(spec: ToolSpec, arguments: dict[str, Any]) -> ToolArgs:
    return spec.args_model.model_validate(arguments)


def resolve_fields(spec: ToolSpec, requested_fields: list[str] | None) -> tuple[str, ...]:
    fields = tuple(requested_fields or spec.default_fields)
    rejected = [field for field in fields if field not in spec.allowed_fields]
    if rejected:
        raise ValueError(f"Unsupported fields for {spec.name}: {', '.join(rejected)}")
    return fields
