#!/usr/bin/env python3
"""
Seed example documents for Saarthi Infotech demo.

Usage:
    # from project root (where app/ exists)
    python scripts/seed_examples.py
"""
import sys
import logging
from pathlib import Path
import json

# ensure project root is on sys.path so we can import app package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services import rag_local_service

logger = logging.getLogger("seed_examples")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

EXAMPLES_DIR = ROOT / "data" / "examples"
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Example documents and associated metadata
EXAMPLES = [
    # =====================================================================
    # 1) HR — Leave & Bonus Policy (Department Confidential)
    # =====================================================================
    {
        "filename": "hr_leave_policy.txt",
        "text": (
            "Saarthi Infotech Leave Policy (example)\n\n"
            "1. Annual Leave: Employees are eligible for 21 days of paid annual leave per year.\n"
            "2. Sick Leave: Employees have 10 days of paid sick leave per year.\n"
            "3. Bonus: Annual bonus is prorated by months worked in the financial year. "
            "Formula: (months_worked / 12) * annual_bonus_amount. For probationary employees, "
            "bonus may be subject to manager approval.\n"
            "4. Approval: Leave must be approved by the manager at least 2 working days in advance for planned leaves.\n"
            "5. Confidentiality: Salary details are confidential and not disclosed by this document.\n"
        ),
        "metadata": {
            "department": "HR",
            "sensitivity": "department_confidential",
            "tags": "leave,bonus,hr_policy",
            "public_summary": (
                "Bonus is prorated based on months worked. Formula: "
                "(months_worked / 12) * annual_bonus_amount."
            )
        },
    },

    # =====================================================================
    # 2) Finance — Expense Reimbursement Policy (Department Confidential)
    # =====================================================================
    {
        "filename": "finance_expense_policy.txt",
        "text": (
            "Saarthi Infotech Expense Reimbursement Policy (example)\n\n"
            "1. Reimbursement: Employees can submit expenses up to INR 50,000 with manager approval.\n"
            "2. Process: Submit expense form, attach receipts, get manager sign-off, then send to Finance for reimbursement.\n"
            "3. Travel: Travel must be booked on approved vendors. International travel requires Finance + Legal pre-approval.\n"
            "4. Timeline: Finance will process approved reimbursements within 30 days.\n"
        ),
        "metadata": {
            "department": "Finance",
            "sensitivity": "department_confidential",
            "tags": "expense,finance,reimbursement",
            "public_summary": (
                "Expenses require receipts and manager sign-off; Finance processes approved "
                "reimbursements within 30 days."
            )
        },
    },

    # =====================================================================
    # 3) HR — Employee Personal Salary Record (Personal Sensitivity)
    # =====================================================================
    {
        "filename": "employee_salary_record_arun.txt",
        "text": (
            "Saarthi Infotech Personal Salary Record (example)\n\n"
            "Employee Name: Arun Sharma\n"
            "Employee ID: emp-101\n"
            "Gross Salary: INR 85,000 per month\n"
            "Allowances: INR 12,000\n"
            "Deductions: INR 5,000\n"
            "This information is strictly confidential and accessible only to HR, Legal, "
            "Executives, and the employee.\n"
        ),
        "metadata": {
            "department": "HR",
            "sensitivity": "personal",
            "owner_id": "emp-101",
            "tags": "salary,personal,hr_record",
            "public_summary": "This is a personal salary document and cannot be shared."
        },
    },

    # =====================================================================
    # 4) Engineering — Onboarding Guide (Public Internal)
    # =====================================================================
    {
        "filename": "engineering_onboarding_guide.txt",
        "text": (
            'Saarthi Infotech Engineering Onboarding Guide (example)\n\n'
            "1. Access Setup: New engineers must request access to Git, Jira, and internal tools.\n"
            "2. Codebase: The primary monorepo uses a microservices architecture.\n"
            "3. Security Keys: Developers must configure SSH keys and complete security training.\n"
            "4. Buddy System: Each new joinee is assigned a mentor.\n"
        ),
        "metadata": {
            "department": "Engineering",
            "sensitivity": "public_internal",
            "tags": "engineering,onboarding,guide",
            "public_summary": (
                "Engineering onboarding includes access setup, development environment "
                "configuration, and codebase introduction."
            )
        },
    },

    # =====================================================================
    # 5) Engineering — System Architecture (Role Confidential)
    # =====================================================================
    {
        "filename": "system_architecture_document.txt",
        "text": (
            "Saarthi Infotech System Architecture Document (example)\n\n"
            "1. Backend: Services communicate using gRPC with load-balanced endpoints.\n"
            "2. Secrets: Stored in HashiCorp Vault with role-based authentication.\n"
            "3. Scaling: Auto-scaling happens on Kubernetes based on CPU and memory thresholds.\n"
            "4. Monitoring: Prometheus + Grafana dashboard pipelines.\n"
        ),
        "metadata": {
            "department": "Engineering",
            "sensitivity": "role_confidential",
            "allowed_roles": ["EngineeringManager", "SeniorEngineer"],
            "tags": "architecture,engineering,confidential",
            "public_summary": (
                "Contains high-level system design details restricted to senior engineering roles."
            )
        },
    },

    # =====================================================================
    # 6) Legal — Vendor Contract (Highly Confidential)
    # =====================================================================
    {
        "filename": "vendor_contract_ABC.txt",
        "text": (
            "Saarthi Infotech Vendor Contract - ABC Pvt Ltd (example)\n\n"
            "1. Term: 24-month exclusive partnership.\n"
            "2. Pricing: Includes volume-based discount tiers.\n"
            "3. Legal Obligations: Penalties for early termination.\n"
            "4. Confidentiality: Strict non-disclosure requirements.\n"
        ),
        "metadata": {
            "department": "Legal",
            "sensitivity": "highly_confidential",
            "tags": "legal,contract,vendor",
            "public_summary": "A confidential legal vendor contract document."
        },
    },

    # =====================================================================
    # 7) Admin — Office Rules (Public Internal)
    # =====================================================================
    {
        "filename": "office_rules_guidelines.txt",
        "text": (
            "Saarthi Infotech Office Rules & Guidelines (example)\n\n"
            "1. Timings: 9:30 AM to 6:30 PM Monday to Friday.\n"
            "2. ID Cards: Must be displayed at all times inside the office premises.\n"
            "3. Dress Code: Smart casuals Monday–Thursday, casual Friday allowed.\n"
        ),
        "metadata": {
            "department": "Admin",
            "sensitivity": "public_internal",
            "tags": "office,rules,admin",
            "public_summary": "General office rules and working hours."
        },
    },

    # =====================================================================
    # 8) IT — Password Reset (Public Internal)
    # =====================================================================
    {
        "filename": "it_password_reset_procedure.txt",
        "text": (
            "Saarthi Infotech IT Password Reset Procedure (example)\n\n"
            "1. Reset Portal: Visit the IT portal and choose 'Reset Password'.\n"
            "2. Verification: Authenticate via OTP emailed to your registered ID.\n"
            "3. Security: Password must follow company complexity rules.\n"
        ),
        "metadata": {
            "department": "IT",
            "sensitivity": "public_internal",
            "tags": "it,password,support",
            "public_summary": "Employees can reset passwords through the IT portal using OTP verification."
        },
    },

    # =====================================================================
    # 9) IT — Network Security (Role Confidential)
    # =====================================================================
    {
        "filename": "network_security_configuration.txt",
        "text": (
            "Saarthi Infotech Network Security Configuration (example)\n\n"
            "1. Firewall: Outbound traffic allowed only on ports 443 and 22.\n"
            "2. VPN: Internal access uses WireGuard-based tunnels.\n"
            "3. Intrusion Detection: Alerts managed by the IT Security team.\n"
        ),
        "metadata": {
            "department": "IT",
            "sensitivity": "role_confidential",
            "allowed_roles": ["ITSecurity", "ITAdmin"],
            "tags": "network,security,it",
            "public_summary": "Network security setup, restricted to IT security roles."
        },
    },
]


def write_examples():
    created = []
    for ex in EXAMPLES:
        p = EXAMPLES_DIR / ex["filename"]
        if not p.exists():
            p.write_text(ex["text"], encoding="utf-8")
            logger.info("Wrote example file: %s", p)
        else:
            logger.info("Example file already exists: %s", p)
        created.append(p)
    return created

def seed_documents():
    # Initialize local RAG (this will setup chroma client / collection)
    try:
        rag_local_service.initialize_local_rag()
    except Exception as e:
        logger.warning("initialize_local_rag raised an exception (may be fine if already initialized): %s", e)

    ingested = {}
    for ex in EXAMPLES:
        path = EXAMPLES_DIR / ex["filename"]
        if not path.exists():
            logger.error("Missing file: %s", path)
            continue
        text = path.read_text(encoding="utf-8")
        try:
            ids = rag_local_service.add_document_to_rag_local(
                source_name=ex["filename"],
                text=text,
                metadata={**ex["metadata"], "ingested_by": "seed_script"}
            )
            ingested[ex["filename"]] = ids
            logger.info("Ingested %s -> %d chunks", ex["filename"], len(ids))
        except Exception as e:
            logger.exception("Failed to ingest %s: %s", ex["filename"], e)
    return ingested

def main():
    logger.info("Seeding example documents to Chroma (local RAG)")
    write_examples()
    ingested = seed_documents()
    if not ingested:
        logger.warning("No documents ingested.")
    else:
        logger.info("Ingestion summary:")
        print(json.dumps(ingested, indent=2))

if __name__ == "__main__":
    main()
