from __future__ import annotations

import unittest

from housing_analytics.acquisition import (
    build_listing_page_url,
    parse_historical_listing_page,
)


SYNTHETIC_LISTING_HTML = """
<ul class="sellListContent">
  <li>
    <div class="title"><a>Invented listing A</a></div>
    <div class="positionInfo"><a>Synthetic community</a><a>Zone A</a></div>
    <div class="houseInfo">2 rooms | 80 sqm</div>
    <div class="followInfo">5 followers | 30 days</div>
    <div class="totalPrice"><span>120</span></div>
    <div class="unitPrice"><span>15,000 CNY/sqm</span></div>
  </li>
  <li>
    <div class="title"><a>Invented listing B</a></div>
    <div class="positionInfo"><a>Synthetic community</a><a>Zone B</a></div>
    <div class="houseInfo">3 rooms | 100 sqm</div>
    <div class="followInfo">2 followers | 60 days</div>
    <div class="totalPrice"><span>180</span></div>
    <div class="unitPrice"><span>18,000 CNY/sqm</span></div>
  </li>
</ul>
"""


class HistoricalAcquisitionEvidenceTests(unittest.TestCase):
    def test_listing_page_url_is_deterministic(self) -> None:
        self.assertEqual(
            build_listing_page_url("https://example-city.lianjia.com/", 7),
            "https://example-city.lianjia.com/ershoufang/pg7/",
        )

    def test_listing_page_url_rejects_unsafe_or_unrelated_hosts(self) -> None:
        with self.assertRaises(ValueError):
            build_listing_page_url("http://example-city.lianjia.com", 1)
        with self.assertRaises(ValueError):
            build_listing_page_url("https://example.com", 1)
        with self.assertRaises(ValueError):
            build_listing_page_url("https://example-city.lianjia.com", 0)

    def test_synthetic_html_reproduces_historical_raw_field_boundary(self) -> None:
        rows = parse_historical_listing_page(
            SYNTHETIC_LISTING_HTML, city_id="SYN_CITY_A"
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].title, "Invented listing A")
        self.assertEqual(rows[0].location_label, "Synthetic community-Zone A")
        self.assertEqual(rows[0].asking_total_price_text, "120")
        self.assertEqual(rows[1].unit_price_text, "18,000 CNY/sqm")


if __name__ == "__main__":
    unittest.main()
