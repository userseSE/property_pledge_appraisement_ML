#!/usr/bin/env python3
"""Generate portfolio charts from datasets under doc/ using only stdlib."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "figures" / "generated"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def load_numeric_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {k: float(v) for k, v in row.items() if k}
            rows.append(parsed)
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def histogram(values: list[float], bins: int, lo: float, hi: float) -> list[int]:
    if hi <= lo:
        return [0] * bins
    counts = [0] * bins
    width = (hi - lo) / bins
    for v in values:
        if v < lo:
            continue
        if v >= hi:
            counts[-1] += 1
            continue
        idx = int((v - lo) / width)
        idx = min(max(idx, 0), bins - 1)
        counts[idx] += 1
    return counts


def pearson(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n == 0:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for i in range(n):
        dx = xs[i] - mx
        dy = ys[i] - my
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    den = math.sqrt(den_x * den_y)
    return 0.0 if den == 0 else num / den


def write_svg(path: Path, width: int, height: int, content: str) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect x="0" y="0" width="100%" height="100%" fill="#0b1220"/>\n'
        f"{content}\n"
        "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def bar_chart(
    out_path: Path,
    labels: list[str],
    values: list[float],
    title: str,
    y_label: str,
    fmt: str = "{:.2f}",
    rotate_labels: bool = False,
) -> None:
    width, height = 1180, 760
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 90, 170 if rotate_labels else 120
    cw = width - margin_left - margin_right
    ch = height - margin_top - margin_bottom
    vmax = max(values) if values else 1.0
    vmax = vmax * 1.1 if vmax > 0 else 1.0
    n = max(len(values), 1)
    gap = 16
    bw = (cw - gap * (n + 1)) / n
    parts: list[str] = []
    parts.append(
        f'<text x="{width/2:.1f}" y="48" text-anchor="middle" '
        'font-family="Arial" font-size="34" fill="#f1f5f9">'
        f"{svg_escape(title)}</text>"
    )
    parts.append(
        f'<text x="26" y="{margin_top + ch/2:.1f}" transform="rotate(-90 26 {margin_top + ch/2:.1f})" '
        'text-anchor="middle" font-family="Arial" font-size="18" fill="#94a3b8">'
        f"{svg_escape(y_label)}</text>"
    )
    # grid and y ticks
    for i in range(6):
        y = margin_top + ch - (ch / 5) * i
        tick_val = vmax * i / 5
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" '
            'stroke="#1f2937" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 6:.1f}" text-anchor="end" '
            'font-family="Arial" font-size="14" fill="#94a3b8">'
            f"{svg_escape(fmt.format(tick_val))}</text>"
        )
    # bars
    for i, (label, val) in enumerate(zip(labels, values)):
        x = margin_left + gap + i * (bw + gap)
        h = 0 if vmax == 0 else (val / vmax) * ch
        y = margin_top + ch - h
        color = "#38bdf8" if i % 2 == 0 else "#22d3ee"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
            f'fill="{color}" rx="4"/>'
        )
        parts.append(
            f'<text x="{x + bw/2:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
            'font-family="Arial" font-size="13" fill="#e2e8f0">'
            f"{svg_escape(fmt.format(val))}</text>"
        )
        if rotate_labels:
            parts.append(
                f'<text x="{x + bw/2:.1f}" y="{margin_top + ch + 24:.1f}" '
                f'transform="rotate(35 {x + bw/2:.1f} {margin_top + ch + 24:.1f})" '
                'text-anchor="start" font-family="Arial" font-size="13" fill="#cbd5e1">'
                f"{svg_escape(label)}</text>"
            )
        else:
            parts.append(
                f'<text x="{x + bw/2:.1f}" y="{margin_top + ch + 24:.1f}" text-anchor="middle" '
                'font-family="Arial" font-size="14" fill="#cbd5e1">'
                f"{svg_escape(label)}</text>"
            )
    write_svg(out_path, width, height, "\n".join(parts))


def overlay_histogram_chart(
    out_path: Path,
    counts_a: list[int],
    counts_b: list[int],
    title: str,
    label_a: str,
    label_b: str,
) -> None:
    width, height = 1180, 760
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 90, 120
    cw = width - margin_left - margin_right
    ch = height - margin_top - margin_bottom
    vmax = max(max(counts_a or [0]), max(counts_b or [0]), 1)
    bins = max(len(counts_a), 1)
    step = cw / max(bins - 1, 1)

    def series_points(counts: list[int]) -> str:
        pts = []
        for i, c in enumerate(counts):
            x = margin_left + i * step
            y = margin_top + ch - (c / vmax) * ch
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    parts: list[str] = []
    parts.append(
        f'<text x="{width/2:.1f}" y="48" text-anchor="middle" '
        'font-family="Arial" font-size="34" fill="#f1f5f9">'
        f"{svg_escape(title)}</text>"
    )
    for i in range(6):
        y = margin_top + ch - (ch / 5) * i
        tick_val = (vmax * i) / 5
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" '
            'stroke="#1f2937" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 6:.1f}" text-anchor="end" '
            'font-family="Arial" font-size="14" fill="#94a3b8">'
            f"{int(tick_val)}</text>"
        )
    parts.append(
        f'<polyline points="{series_points(counts_a)}" fill="none" stroke="#38bdf8" stroke-width="3"/>'
    )
    parts.append(
        f'<polyline points="{series_points(counts_b)}" fill="none" stroke="#f97316" stroke-width="3"/>'
    )
    parts.append(
        f'<text x="{margin_left}" y="{height - 28}" font-family="Arial" font-size="15" fill="#38bdf8">'
        f"{svg_escape(label_a)}</text>"
    )
    parts.append(
        f'<text x="{margin_left + 180}" y="{height - 28}" font-family="Arial" font-size="15" fill="#f97316">'
        f"{svg_escape(label_b)}</text>"
    )
    write_svg(out_path, width, height, "\n".join(parts))


def read_describe3_means(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) < 3:
        return {}
    headers = rows[0][1:]
    mean_row = rows[2][1:]
    out: dict[str, float] = {}
    for k, v in zip(headers, mean_row):
        if not k:
            continue
        out[k] = float(v)
    return out


def main() -> None:
    ensure_dir(OUT_DIR)

    train_path = ROOT / "doc" / "DataSet" / "training" / "train_data_second.csv"
    test_path = ROOT / "doc" / "DataSet" / "training" / "test_data_second.csv"
    describe3_path = ROOT / "doc" / "描述统计" / "describe3.csv"

    train_rows = load_numeric_rows(train_path)
    test_rows = load_numeric_rows(test_path)

    # 1) Train/test row count
    bar_chart(
        OUT_DIR / "doc_train_test_split_counts.svg",
        labels=["Train", "Test"],
        values=[float(len(train_rows)), float(len(test_rows))],
        title="Train/Test Split Size (from doc/DataSet/training)",
        y_label="Row Count",
        fmt="{:.0f}",
    )

    # 2) Target distribution histogram
    train_target = [r["t_price"] for r in train_rows]
    test_target = [r["t_price"] for r in test_rows]
    cap = max(percentile(train_target + test_target, 0.99), 100.0)
    bins = 42
    train_hist = histogram(train_target, bins=bins, lo=0.0, hi=cap)
    test_hist = histogram(test_target, bins=bins, lo=0.0, hi=cap)
    overlay_histogram_chart(
        OUT_DIR / "doc_t_price_distribution_train_vs_test.svg",
        train_hist,
        test_hist,
        title=f"Target Distribution (t_price, capped at P99={cap:.1f})",
        label_a="Train Histogram",
        label_b="Test Histogram",
    )

    # 3) Top feature correlation with target from training split
    headers = list(train_rows[0].keys()) if train_rows else []
    features = [h for h in headers if h != "t_price"]
    y = [r["t_price"] for r in train_rows]
    corr_pairs: list[tuple[str, float]] = []
    for feature in features:
        x = [r[feature] for r in train_rows]
        corr_pairs.append((feature, abs(pearson(x, y))))
    corr_pairs.sort(key=lambda kv: kv[1], reverse=True)
    top = corr_pairs[:10]
    bar_chart(
        OUT_DIR / "doc_top10_abs_corr_with_t_price.svg",
        labels=[k for k, _ in top],
        values=[v for _, v in top],
        title="Top 10 |Correlation| with t_price (Train Split)",
        y_label="Absolute Pearson Correlation",
        fmt="{:.3f}",
        rotate_labels=True,
    )

    # 4) Binary tag rates from describe3 mean row
    means = read_describe3_means(describe3_path)
    tag_cols = ["park", "light", "p_lot", "water", "business", "transport", "l_tax"]
    tag_vals = [means.get(c, 0.0) * 100 for c in tag_cols]
    bar_chart(
        OUT_DIR / "doc_binary_tag_mean_rates.svg",
        labels=tag_cols,
        values=tag_vals,
        title="Binary Feature Mean Rates (describe3 mean row)",
        y_label="Rate (%)",
        fmt="{:.1f}",
    )

    print("Generated charts:")
    for p in sorted(OUT_DIR.glob("*.svg")):
        print(f"- {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

