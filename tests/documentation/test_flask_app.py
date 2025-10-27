"""Integration tests for the Flask item endpoint."""

from __future__ import annotations

from http import HTTPStatus

import pytest

from tasks.documentation.flask_app import Item, create_app


@pytest.fixture()
def client():
    """Provide a Flask test client configured with deterministic fixtures."""

    app = create_app({1: Item(id=1, name="Laptop", price=1999.99)})
    with app.test_client() as test_client:
        yield test_client


def test_get_item_success(client):
    response = client.get("/items/1")
    assert response.status_code == HTTPStatus.OK
    validated = Item.model_validate_json(response.data)
    assert validated.name == "Laptop"
    assert validated.price == pytest.approx(1999.99)


def test_get_item_not_found(client):
    response = client.get("/items/999")
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert b"Item not found" in response.data


def test_schema_violation_logs_error(monkeypatch, caplog):
    class BrokenItem(Item):
        price: float

    captured = BrokenItem(id=5, name="Broken", price=10.0)

    def _failing_validate(item):  # type: ignore[override]
        raise ValueError("boom")

    monkeypatch.setattr(
        "tasks.documentation.flask_app.Item.model_validate",
        staticmethod(_failing_validate),
    )
    app = create_app({5: captured})
    with app.test_client() as test_client:
        caplog.clear()
        caplog.set_level("ERROR")
        response = test_client.get("/items/5")

    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert any(
        "Item payload failed validation" in record.message for record in caplog.records
    )
