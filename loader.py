"""
Regulatory document loader with live-fetch fallback.
"""

import re
import logging
from datetime import date
from typing import Dict, List

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("ComplianceAdvisor")


class RegulatoryDocumentLoader:
    """
    Loads real KRA/RRA regulatory texts with fallback to high-fidelity stubs.
    """

    SOURCES: List[Dict] = [
        {
            "url": "https://www.kra.go.ke/individual/filing-paying/types-of-taxes/value-added-tax",
            "jurisdiction": "Kenya",
            "tax_category": "VAT",
            "citation": "VAT Act Cap 476",
            "source_url": "https://www.kra.go.ke",
        },
        {
            "url": "https://www.kra.go.ke/individual/filing-paying/types-of-taxes/pay-as-you-earn",
            "jurisdiction": "Kenya",
            "tax_category": "PAYE",
            "citation": "Income Tax Act Cap 470",
            "source_url": "https://www.kra.go.ke",
        },
        {
            "url": "https://www.rra.gov.rw/index.php?id=474",
            "jurisdiction": "Rwanda",
            "tax_category": "VAT",
            "citation": "Law No. 37/2012",
            "source_url": "https://www.rra.gov.rw",
        },
        {
            "url": "https://www.rra.gov.rw/index.php?id=472",
            "jurisdiction": "Rwanda",
            "tax_category": "PAYE",
            "citation": "Income Tax Law No. 16/2018",
            "source_url": "https://www.rra.gov.rw",
        },
    ]

    FALLBACK_DOCS: List[Dict] = [
        {
            "content": """KENYA REVENUE AUTHORITY – VALUE ADDED TAX (VAT)
Source: VAT Act Cap 476, as amended by Finance Act 2023

REGISTRATION
A person is required to apply for VAT registration when their taxable turnover
exceeds KES 5,000,000 in any consecutive 12-month period (Section 5, VAT Act).
Application must be made within 30 days of reaching the threshold via iTax.

RATES
• Standard rate: 16% on taxable supplies (Section 5(2))
• Zero rate (0%): exports, certain food items, medicaments (Second Schedule)
• Exempt: financial services, residential rent (First Schedule)

FILING & PAYMENT
Returns: Monthly by the 20th day of the following month on iTax.
Payment: Same deadline as filing.

PENALTIES
Late filing:  KES 10,000 or 5% of tax due (whichever is higher) per Section 83
Late payment: 20% of unpaid tax + interest at the prevailing CBK rate per annum

INPUT TAX CREDITS
Registered taxpayers may claim input tax on business purchases within 6 months
of the tax invoice date. Blocked credits: passenger vehicles, entertainment.""",
            "metadata": {
                "jurisdiction": "Kenya",
                "tax_category": "VAT",
                "citation": "VAT Act Cap 476, Finance Act 2023",
                "source_url": "https://www.kra.go.ke",
                "last_updated": "2024-07-01",
            },
        },
        {
            "content": """KENYA REVENUE AUTHORITY – PAY AS YOU EARN (PAYE)
Source: Income Tax Act Cap 470, as amended by Finance Act 2023

OBLIGATION
Every employer paying employment income exceeding the personal relief threshold
must deduct and remit PAYE (Section 37, ITA).

PERSONAL RELIEF: KES 28,800 per annum (KES 2,400/month) from 1 July 2023.

TAX BANDS (w.e.f. 1 July 2023, monthly)
  KES 0       – 24,000    → 10%
  KES 24,001  – 32,333    → 25%
  KES 32,334  – 500,000   → 30%
  KES 500,001 – 800,000   → 32.5%
  Above KES 800,000       → 35%

HOUSING LEVY (Finance Act 2023)
1.5% of gross salary deducted by employer; employer contributes 1.5%.
Remitted together with PAYE.

FILING & PAYMENT
Returns: 9th day of the following month via iTax.
Payment: Same deadline.

PENALTIES
Late remittance: 25% of unpaid tax + CBK rate interest.""",
            "metadata": {
                "jurisdiction": "Kenya",
                "tax_category": "PAYE",
                "citation": "Income Tax Act Cap 470, Finance Act 2023",
                "source_url": "https://www.kra.go.ke",
                "last_updated": "2024-07-01",
            },
        },
        {
            "content": """RWANDA REVENUE AUTHORITY – VALUE ADDED TAX (TVA)
Source: Law No. 37/2012 of 09/11/2012 on Value Added Tax (as amended)

REGISTRATION THRESHOLD
Any person whose annual taxable turnover exceeds RWF 20,000,000 must register
for VAT within 7 days of reaching the threshold (Article 4).

RATES
• Standard rate: 18% (Article 5)
• Zero rate:     Exports, international transport (Article 6)
• Exempt:        Agricultural inputs, financial services (Article 7)

FILING & PAYMENT
Monthly return: 15th day of the month following the tax period.
Annual return:  31 January of the following year.
All filings via M-Declaration / E-Tax portal (https://etax.rra.gov.rw).

PENALTIES (Article 57, Tax Procedures Law No. 16/2018)
Late filing:  RWF 100,000 per month (uncapped)
Late payment: 1.5% interest per month on outstanding amount

INVOICING
VAT-registered taxpayers must issue Electronic Billing Machine (EBM) receipts
for every supply (Ministerial Order No. 004/16/10/TC of 25/09/2013).""",
            "metadata": {
                "jurisdiction": "Rwanda",
                "tax_category": "VAT",
                "citation": "Law No. 37/2012, Tax Procedures Law No. 16/2018",
                "source_url": "https://www.rra.gov.rw",
                "last_updated": "2024-01-15",
            },
        },
        {
            "content": """RWANDA REVENUE AUTHORITY – PAY AS YOU EARN (PAYE)
Source: Income Tax Law No. 16/2018 of 13/04/2018

OBLIGATION
Every employer must withhold income tax from employees' remuneration and remit
it to RRA (Article 56, ITL).

TAX BANDS (monthly, Article 9)
  RWF 0         – 30,000    → 0%   (exempt band)
  RWF 30,001    – 100,000   → 20%
  Above RWF 100,000         → 30%

BENEFITS IN KIND
Taxable at fair market value; vehicles: 10% of cost per month.

FILING & PAYMENT
Monthly return and payment: 15th day of the following month via E-Tax portal.

PENALTIES (Tax Procedures Law No. 16/2018)
Late filing:  RWF 300,000 flat fine
Late payment: 1.5% interest per month on unpaid tax

SOCIAL SECURITY
Pension (RSSB): Employee 3%, Employer 5% of gross salary.
CBHI (Mutuelle): Employee 0.5%, Employer 0.5%.""",
            "metadata": {
                "jurisdiction": "Rwanda",
                "tax_category": "PAYE",
                "citation": "Income Tax Law No. 16/2018, Tax Procedures Law No. 16/2018",
                "source_url": "https://www.rra.gov.rw",
                "last_updated": "2024-01-15",
            },
        },
    ]

    def load(self, use_live_fetch: bool = False) -> List[Dict]:
        """If use_live_fetch=False (default), return the high-fidelity stubs directly."""
        if not use_live_fetch:
            return self.FALLBACK_DOCS

        docs: List[Dict] = []
        for src in self.SOURCES:
            doc = self._fetch(src, timeout=8)
            docs.append(doc)
        return docs

    def _fetch(self, src: Dict, timeout: int) -> Dict:
        try:
            resp = requests.get(
                src["url"],
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["nav", "footer", "script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)[:8000]
            logger.info(f"✓ Live fetch: {src['url']}")
            return {
                "content": text,
                "metadata": {
                    **{k: v for k, v in src.items() if k != "url"},
                    "last_updated": date.today().isoformat(),
                    "fetched_live": True,
                },
            }
        except Exception as e:
            logger.warning(
                f"Live fetch failed ({e}); using fallback stub for "
                f"{src['jurisdiction']} {src['tax_category']}"
            )
            for stub in self.FALLBACK_DOCS:
                if (
                    stub["metadata"]["jurisdiction"] == src["jurisdiction"]
                    and stub["metadata"]["tax_category"] == src["tax_category"]
                ):
                    return stub
            raise RuntimeError(f"No fallback for {src}")
