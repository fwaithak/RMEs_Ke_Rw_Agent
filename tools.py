"""
Mock tool APIs for document automation and scheduling.

In production, these would call real external services.
"""

import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ComplianceAdvisor")


def document_automation_api(
    jurisdiction: str,
    compliance_type: str,
    retrieved_docs: List[Dict],
    business_profile: Dict,
) -> Dict[str, Any]:
    """
    Mock document automation tool.

    Generates a jurisdiction-specific compliance checklist from
    retrieved regulatory documents.
    """
    checklist: Dict[str, Any] = {
        "jurisdiction": jurisdiction,
        "compliance_type": compliance_type,
        "steps": [],
        "required_documents": [],
        "deadlines": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "business_type": business_profile.get("business_type", "unknown"),
        "turnover_tier": business_profile.get("turnover_tier", "unknown"),
    }

    if not retrieved_docs:
        checklist["steps"] = [
            "No documents retrieved — cannot generate checklist."
        ]
        return checklist

    all_content = " ".join(d.get("content", "") for d in retrieved_docs)
    checklist["steps"] = [
        f"1. Confirm {compliance_type} registration status with the "
        f"{jurisdiction} tax authority.",
        "2. Collect all required records for the filing period.",
        f"3. Complete the {compliance_type} return form.",
        "4. Calculate tax liability and reconcile with accounts.",
        "5. Submit the return and make payment by the deadline.",
        "6. Retain submission confirmation for your records.",
    ]
    checklist["required_documents"] = [
        "Tax Identification Number (TIN / PIN)",
        "Business registration certificate",
        f"{compliance_type} invoices (sales and purchases) for the period",
        "Bank statements",
    ]

    deadline_patterns = [
        r"(\d{1,2})(?:st|nd|rd|th)?\s+day\s+of",
        r"by\s+the\s+(\d{1,2})(?:st|nd|rd|th)?",
        r"before\s+the\s+(\d{1,2})(?:st|nd|rd|th)?",
        r"on\s+(?:or\s+before\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(?:day|of)",
    ]
    deadline_day: Optional[str] = None
    try:
        for pattern in deadline_patterns:
            match = re.search(pattern, all_content, re.IGNORECASE)
            if match:
                deadline_day = match.group(1)
                break
    except Exception as exc:
        logger.warning(f"Deadline regex failed: {exc}")

    if deadline_day:
        day_int = int(deadline_day)
        suffix = (
            "th" if 11 <= day_int <= 13
            else {1: "st", 2: "nd", 3: "rd"}.get(day_int % 10, "th")
        )
        checklist["deadlines"] = [
            f"Monthly {compliance_type} filing due on the "
            f"{deadline_day}{suffix} of each month."
        ]
    else:
        checklist["deadlines"] = [
            f"{compliance_type} filing deadline: check the "
            f"{jurisdiction} revenue authority website."
        ]

    return checklist


def productivity_scheduler_api(
    event_type: str,
    jurisdiction: str,
    deadline_date: Optional[str],
    recurrence: Optional[str],
    user_contact: Dict,
) -> Dict[str, Any]:
    """
    Mock scheduling tool.

    Creates a calendar event and email reminder.
    """
    if not deadline_date:
        deadline_date = (
            datetime.now(timezone.utc) + timedelta(days=30)
        ).date().isoformat()

    return {
        "calendar_event_id": f"evt_{uuid.uuid4().hex[:8]}",
        "notification_status": True,
        "scheduled_timestamp": datetime.now(timezone.utc).isoformat(),
        "event_details": {
            "title": f"{event_type} Reminder – {jurisdiction}",
            "date": deadline_date,
            "recurrence": recurrence,
        },
        "notification_sent_to": user_contact.get("email", "unknown"),
    }
