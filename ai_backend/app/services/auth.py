# app/services/auth.py

_API_KEYS = {
    # ============================================================
    # 🟦 GENERAL EMPLOYEES (Regular Users)
    # ============================================================
    "key-employee-1": {
        "user_id": "u_emp_1",
        "role": "Employee",
        "department": "Engineering",
    },
    "key-employee-2": {
        "user_id": "u_emp_2",
        "role": "Employee",
        "department": "Finance",
    },
    "key-employee-3": {
        "user_id": "u_emp_3",
        "role": "Employee",
        "department": "HR",
    },

    # ============================================================
    # 🟦 ENGINEERING ROLES
    # ============================================================
    "key-engineer-1": {
        "user_id": "u_eng_1",
        "role": "Engineer",
        "department": "Engineering",
    },
    "key-senior-engineer-1": {
        "user_id": "u_seng_1",
        "role": "SeniorEngineer",
        "department": "Engineering",
    },
    "key-engineering-manager-1": {
        "user_id": "u_engmgr_1",
        "role": "EngineeringManager",
        "department": "Engineering",
    },

    # ============================================================
    # 🟧 MANAGERS
    # ============================================================
    "key-manager-1": {
        "user_id": "u_mgr_1",
        "role": "Manager",
        "department": "Engineering",
    },
    "key-manager-2": {
        "user_id": "u_mgr_2",
        "role": "Manager",
        "department": "Finance",
    },

    # ============================================================
    # 🟪 HR TEAM
    # ============================================================
    "key-hr-1": {
        "user_id": "u_hr_1",
        "role": "HR",
        "department": "HR",
    },
    "key-hr-manager-1": {
        "user_id": "u_hrmgr_1",
        "role": "HRManager",
        "department": "HR",
    },

    # ============================================================
    # 🟥 LEGAL TEAM
    # ============================================================
    "key-legal-1": {
        "user_id": "u_legal_1",
        "role": "Legal",
        "department": "Legal",
    },
    "key-legal-contract-1": {
        "user_id": "u_legal_contract_1",
        "role": "LegalAdvisor",
        "department": "Legal",
    },

    # ============================================================
    # 🟩 FINANCE TEAM
    # ============================================================
    "key-finance-1": {
        "user_id": "u_fin_1",
        "role": "FinanceAssociate",
        "department": "Finance",
    },
    "key-finance-manager-1": {
        "user_id": "u_finmgr_1",
        "role": "FinanceManager",
        "department": "Finance",
    },

    # ============================================================
    # 🟨 IT & SECURITY ROLES
    # ============================================================
    "key-it-1": {
        "user_id": "u_it_1",
        "role": "ITSupport",
        "department": "IT",
    },
    "key-it-admin-1": {
        "user_id": "u_itadmin_1",
        "role": "ITAdmin",
        "department": "IT",
    },
    "key-it-security-1": {
        "user_id": "u_itsec_1",
        "role": "ITSecurity",
        "department": "IT",
    },

    # ============================================================
    # 🟫 EXECUTIVE / LEADERSHIP TEAM
    # ============================================================
    "key-exec-1": {
        "user_id": "u_exec_1",
        "role": "Executive",
        "department": "Executive",
    },
    "key-ceo-1": {
        "user_id": "u_ceo_1",
        "role": "CEO",
        "department": "Executive",
    },
    "key-cto-1": {
        "user_id": "u_cto_1",
        "role": "CTO",
        "department": "Executive",
    },

    # ============================================================
    # ⬜ GUEST / TEMP ACCESS
    # ============================================================
    "key-guest-1": {
        "user_id": "u_guest_1",
        "role": "Guest",
        "department": "General",
    },
}


def get_user_from_api_key(key: str):
    return _API_KEYS.get(key)
