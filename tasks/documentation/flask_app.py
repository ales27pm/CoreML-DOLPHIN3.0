"""Flask application exposing an item catalogue endpoint with validation."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from logging import getLogger
from typing import Mapping, MutableMapping

from flask import Flask, abort, jsonify  # type: ignore[import-not-found,import-untyped]
from pydantic import BaseModel, ValidationError

LOGGER = getLogger(__name__)


class Item(BaseModel):
    """Serializable item representation."""

    id: int
    name: str
    price: float


@dataclass(slots=True)
class Inventory:
    """In-memory item catalogue with validation hooks."""

    _records: MutableMapping[int, Item]

    @classmethod
    def from_mapping(cls, mapping: Mapping[int, Item] | None = None) -> "Inventory":
        """Construct an inventory from a mapping, falling back to defaults."""

        seed = mapping or {1: Item(id=1, name="Laptop", price=1999.99)}
        return cls(_records=dict(seed))

    def get(self, item_id: int) -> Item | None:
        """Retrieve an item by identifier."""

        return self._records.get(item_id)

    def upsert(self, item: Item) -> None:
        """Insert or replace an item entry with schema validation."""

        self._records[item.id] = Item.model_validate(item)


def create_app(initial_data: Mapping[int, Item] | None = None) -> Flask:
    """Create a configured Flask application instance.

    The returned application exposes a single `/items/<int:item_id>` endpoint that
    validates outgoing payloads with Pydantic before serialising them to JSON.
    """

    app = Flask(__name__)
    inventory = Inventory.from_mapping(initial_data)

    @app.get("/items/<int:item_id>")
    def get_item(item_id: int):
        item = inventory.get(item_id)
        if item is None:
            LOGGER.info("Requested missing item", extra={"item_id": item_id})
            abort(HTTPStatus.NOT_FOUND, description="Item not found")

        try:
            payload = Item.model_validate(item).model_dump()
        except (ValidationError, ValueError) as exc:
            LOGGER.exception(
                "Item payload failed validation",
                extra={"item_id": item_id},
            )
            abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(exc))
        return jsonify(payload)

    return app


__all__ = ["Item", "Inventory", "create_app"]
