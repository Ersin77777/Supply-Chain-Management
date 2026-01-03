# Data (Not Included)
The original datasets used in this project are **confidential** and therefore
**not included** in this repository.

---
## 📁 Expected File Locations
Place your CSV files in the following paths:
- `data/raw/company_a.csv`
- `data/raw/company_b.csv`
These paths are assumed by default in the notebooks and source code.

---
## 📄 Expected File Format
- File type: CSV
- Delimiter: Semicolon (`;`)
- Encoding: UTF-8 (recommended)

---
## 🧱 Required Columns (Minimum)
Each dataset must contain the following columns:
- `Planned Delivery Date`
- `Arrival Date`
- `Ordered Quantity`
- `Delivered Quantity`
- `Supplier`

---
## ➕ Optional Columns
If available, the following columns will be used for extended analysis
and modeling:
- `Product Article Number`
- `Quality` (Company A only)

---
## ⚠️ Data Assumptions & Business Rules
The pipeline applies the following assumptions:
- `Arrival Date` may be missing for open or undelivered orders.
- `Ordered Quantity = 0` is treated as an invalid order.
- Extreme or implausible planned dates are flagged as anomalies.
- Weekend and public holiday effects are derived from the planned delivery date.
- Early deliveries are tracked but not considered equivalent to on-time delivery.

---

## 🔒 Data Privacy
This repository is intended for use with **anonymized procurement data** only.

Users are responsible for ensuring that all datasets comply with
internal confidentiality, legal, and data protection requirements.