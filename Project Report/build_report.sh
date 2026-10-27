#!/usr/bin/env bash
# Builds WildGuard_Final_Report.docx and .pdf from WildGuard_Final_Report.md.
# Needs python3, pandoc and LibreOffice (soffice).
set -euo pipefail
cd "$(dirname "$0")"
cp ../"SYSTEM CODE"/outputs/figures/*.png figures/
python3 prep_tables.py WildGuard_Final_Report.md .report_build.md
pandoc .report_build.md -o WildGuard_Final_Report.docx --resource-path=. \
  --reference-doc=reference.docx --columns=100
rm -f .report_build.md
soffice --headless --convert-to pdf WildGuard_Final_Report.docx >/dev/null
echo "built WildGuard_Final_Report.docx and .pdf"
