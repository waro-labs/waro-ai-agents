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
    domain: Literal["sales", "food_cost", "menu", "financial"]
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
        default_fields=("id", "name", "price", "is_available"),
        allowed_fields=frozenset(
            {"id", "name", "price", "is_available", "category", "cost", "margin"}
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
        default_fields=("id", "name", "is_active", "cost"),
        allowed_fields=frozenset({"id", "name", "is_active", "cost", "ingredients_count"}),
        domain="menu",
        description="List menu recipes and recipe cost/activity metadata.",
        tags=("menu", "recipes", "ingredients", "cost"),
        examples=("recetas activas", "costos de recetas"),
    ),
    "waro.sales.list": ToolSpec(
        name="waro.sales.list",
        command=("sales", "list"),
        scope="orders:read",
        args_model=SalesListArgs,
        default_fields=("id", "status", "total", "order_date"),
        allowed_fields=frozenset(
            {"id", "status", "total", "order_date", "payment_method", "customer_name"}
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
        default_fields=("id", "name", "margin", "revenue", "cost"),
        allowed_fields=frozenset({"id", "name", "margin", "revenue", "cost", "quantity"}),
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
