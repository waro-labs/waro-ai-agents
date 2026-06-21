from app.tools.data_shaper import DataShaper, ShapeProfile


def test_data_shaper_filters_ranks_and_limits_rows():
    shaper = DataShaper()
    profile = ShapeProfile(
        entity="customers",
        allowed_rank_fields=frozenset({"order_count", "total_spent"}),
        default_rank_fields=("total_spent", "order_count"),
        active_fields=("order_count", "total_spent"),
    )

    result = shaper.shape_ranked_rows(
        [
            {"name": "Inactive", "order_count": 0, "total_spent": 0},
            {"name": "High spend", "order_count": 1, "total_spent": 300000},
            {"name": "Frequent", "order_count": 4, "total_spent": 120000},
        ],
        profile=profile,
        operations=[
            {"type": "rank", "by": ["order_count", "total_spent"], "direction": "desc"},
            {"type": "limit", "value": 2},
        ],
        limit=2,
    )

    assert [row["name"] for row in result.rows] == ["Frequent", "High spend"]
    assert result.execution["filtered_rows"] == 1
    assert result.execution["sort_fields"] == ["order_count", "total_spent"]
    assert result.execution["limit"] == 2


def test_data_shaper_uses_default_rank_operations_when_missing():
    shaper = DataShaper()
    profile = ShapeProfile(
        entity="products",
        allowed_rank_fields=frozenset({"quantity", "revenue"}),
        default_rank_fields=("quantity", "revenue"),
        active_fields=("quantity", "revenue"),
    )

    result = shaper.shape_ranked_rows(
        [{"name": "A", "quantity": 0}, {"name": "B", "quantity": 3}],
        profile=profile,
        operations=None,
        limit=10,
    )

    assert [row["name"] for row in result.rows] == ["B"]
    assert result.execution["operations"] == [
        {"type": "filter", "condition": "quantity > 0 OR revenue > 0"},
        {"type": "rank", "by": ["quantity", "revenue"], "direction": "desc"},
        {"type": "limit", "value": 10},
    ]


def test_data_shaper_ignores_unknown_rank_fields():
    shaper = DataShaper()
    profile = ShapeProfile(
        entity="customers",
        allowed_rank_fields=frozenset({"order_count", "total_spent"}),
        default_rank_fields=("total_spent", "order_count"),
    )

    fields = shaper.rank_fields(
        profile=profile,
        operations=[{"type": "rank", "by": ["data", "total_spent"]}],
        sort_field=None,
    )

    assert fields == ("total_spent",)
