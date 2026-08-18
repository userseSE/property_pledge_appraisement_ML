"""Sanitized evidence of the historical second-hand listing acquisition step.

The original 2022 notebook discovered city sites, paginated ``/ershoufang/``
pages, and parsed listing-card fields. This module preserves the deterministic
URL and HTML parsing boundary without making network requests, rotating user
agents, using proxies, or writing third-party rows to the repository.

It documents historical acquisition; it is not a claim that the current site
can still be collected with these selectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class HistoricalListingRow:
    """One listing card as captured before the legacy cleaning stage."""

    city_id: str
    title: str
    location_label: str
    house_info: str
    follow_info: str
    asking_total_price_text: str
    unit_price_text: str


def build_listing_page_url(city_base_url: str, page_number: int) -> str:
    """Build the historical Lianjia second-hand listing pagination URL."""

    if isinstance(page_number, bool) or not isinstance(page_number, int):
        raise TypeError("page_number must be an integer")
    if page_number < 1:
        raise ValueError("page_number must be at least 1")

    parsed = urlsplit(city_base_url.strip())
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        raise ValueError("city_base_url must use HTTPS")
    if hostname != "lianjia.com" and not hostname.endswith(".lianjia.com"):
        raise ValueError("city_base_url must be a Lianjia city host")
    if parsed.username or parsed.password or parsed.port:
        raise ValueError("city_base_url must not contain credentials or a custom port")

    return urlunsplit(
        ("https", parsed.netloc.lower(), f"/ershoufang/pg{page_number}/", "", "")
    )


def _classes(attributes: list[tuple[str, str | None]]) -> frozenset[str]:
    class_value = next((value for key, value in attributes if key == "class"), "")
    return frozenset((class_value or "").split())


def _normalized_text(parts: list[str], *, separator: str = " ") -> str:
    cleaned = [" ".join(part.split()) for part in parts]
    return separator.join(part for part in cleaned if part)


class _HistoricalListingParser(HTMLParser):
    field_by_class = {
        "title": "title",
        "positionInfo": "location_label",
        "houseInfo": "house_info",
        "followInfo": "follow_info",
        "totalPrice": "asking_total_price_text",
        "unitPrice": "unit_price_text",
    }

    def __init__(self, city_id: str) -> None:
        super().__init__(convert_charrefs=True)
        self.city_id = city_id
        self.rows: list[HistoricalListingRow] = []
        self._elements: list[tuple[str, frozenset[str]]] = []
        self._results_depth: int | None = None
        self._row_depth: int | None = None
        self._current: dict[str, list[str]] | None = None
        self._captures: list[tuple[int, str]] = []

    def handle_starttag(
        self, tag: str, attributes: list[tuple[str, str | None]]
    ) -> None:
        classes = _classes(attributes)
        self._elements.append((tag, classes))
        depth = len(self._elements)

        if self._results_depth is None and "sellListContent" in classes:
            self._results_depth = depth
        elif (
            self._results_depth is not None
            and self._current is None
            and tag == "li"
            and depth > self._results_depth
        ):
            self._row_depth = depth
            self._current = {
                field: [] for field in self.field_by_class.values()
            }

        if self._current is not None:
            for class_name, field_name in self.field_by_class.items():
                if class_name in classes:
                    self._captures.append((depth, field_name))
                    break

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._captures:
            self._current[self._captures[-1][1]].append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._elements:
            return
        depth = len(self._elements)
        if self._captures and self._captures[-1][0] == depth:
            self._captures.pop()

        if self._current is not None and self._row_depth == depth and tag == "li":
            self.rows.append(
                HistoricalListingRow(
                    city_id=self.city_id,
                    title=_normalized_text(self._current["title"]),
                    location_label=_normalized_text(
                        self._current["location_label"], separator="-"
                    ),
                    house_info=_normalized_text(self._current["house_info"]),
                    follow_info=_normalized_text(self._current["follow_info"]),
                    asking_total_price_text=_normalized_text(
                        self._current["asking_total_price_text"]
                    ),
                    unit_price_text=_normalized_text(
                        self._current["unit_price_text"]
                    ),
                )
            )
            self._current = None
            self._row_depth = None
            self._captures.clear()

        if self._results_depth == depth and "sellListContent" in self._elements[-1][1]:
            self._results_depth = None
        self._elements.pop()


def parse_historical_listing_page(
    html_text: str, *, city_id: str
) -> tuple[HistoricalListingRow, ...]:
    """Parse historical listing-card fields from already acquired HTML."""

    normalized_city = " ".join(city_id.split())
    if not normalized_city:
        raise ValueError("city_id must not be blank")
    parser = _HistoricalListingParser(normalized_city)
    parser.feed(html_text)
    parser.close()
    return tuple(parser.rows)
