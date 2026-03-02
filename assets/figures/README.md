# Figure Sources

## Generated Charts

- Folder: `assets/figures/generated/`
- Script: `scripts/generate_portfolio_charts.py`
- Data sources:
  - `doc/DataSet/training/train_data_second.csv`
  - `doc/DataSet/training/test_data_second.csv`
  - `doc/描述统计/describe3.csv`

Regenerate:

```bash
python3 scripts/generate_portfolio_charts.py
```

## Extracted Charts From Word

- Folder: `assets/figures/docx_selected/`
- Extracted from:
  - `doc/基于机器学习的商业银行房产质押价值评估研究.docx`
- Raw extraction folder:
  - `assets/figures/docx_extracted/word/media/`

If privacy is a concern, keep only `docx_selected/` and remove `docx_extracted/` after selecting needed figures.
