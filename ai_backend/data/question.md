Here is a set of **Verification Questions and Answers** based on the 21 files I provided.

Use these to test your RAG app. If the app answers correctly, it proves it is reading your specific files, as these details do not exist on the public internet.

---

### Category 1: HR & Policy (Simple Retrieval)

**Q1: What is the company policy regarding "Wellness Days"?**
*   **Answer:** Employees are entitled to **4 "Unplug" days per year** (1 per quarter). This was mandated by CEO Aisha Sharma to prevent burnout.
*   **Source:** `HR_policies_handbook.md`

**Q2: I am traveling to New York for business. What is my hotel budget?**
*   **Answer:** New York is considered a Tier 1 city, so the accommodation cap is **$250 USD per night**.
*   **Source:** `expense_reimbursement_policy.md`

---

### Category 2: IT & Security (Specific Constraints)

**Q3: What are the requirements for a new password?**
*   **Answer:** It must be at least **14 characters long**, contain Uppercase, Lowercase, Number, and Special Character, and must be rotated every **90 days**.
*   **Source:** `IT_security_policy.md`

**Q4: Can I save "Level 4" pharmaceutical data on my USB drive?**
*   **Answer:** **No.** Level 4 (Restricted) data cannot be stored on local drives or USB sticks. It must remain in the secure 'Vajra-Vault' cloud storage.
*   **Source:** `IT_security_policy.md`

---

### Category 3: Business Logic & Projects (Synthesis)

**Q5: What was the result of "Project Surya"?**
*   **Answer:** The project reduced the clinical trial phase from 12 months to **7 months**, saved **$4.5 Million USD**, and helped the drug *CardioFix-X* get approved ahead of schedule.
*   **Source:** `project_case_study_surya.md`

**Q6: What is the "Green Sky" project proposal?**
*   **Answer:** It is a proposal by Praxis Global Logistics to use **autonomous hexacopter drones** for delivering medicines in the Himalayan regions. It is requesting a budget of **$15 Million**.
*   **Source:** `project_proposal_green_sky.md`

---

### Category 4: Technical & Incidents (Detail Recall)

**Q7: What caused the outage on August 14, 2023 (INC-402)?**
*   **Answer:** It was caused by a **misconfigured Terraform script** (deployed by a Junior Engineer) that set the Load Balancer `max_connections` to **0**.
*   **Source:** `technical_incident_postmortem_INC-402.md`

**Q8: I want to use the Thunderbolt API to predict sales. What is the endpoint?**
*   **Answer:** The endpoint is `POST /api/v1/forecast/sales`. You must include the `X-Agni-Auth-Token` header.
*   **Source:** `api_documentation_thunderbolt_v1.md`

---

### Category 5: People & Culture (Entity Extraction)

**Q9: Who is Kenji Sato and what is his hobby?**
*   **Answer:** Kenji Sato is the **Group COO** based in Tokyo. His hobby is **marathon running** (he has completed the Tokyo Marathon 4 times) and Tea Ceremonies.
*   **Source:** `executive_bio_kenji_sato.md`

**Q10: Who won the Employee of the Month in November 2023?**
*   **Answer:** **Sarah Jenkins** from Nectar Digital (London office) for her work on the "Winter Campaign".
*   **Source:** `internal_newsletter_the_spark_nov23.md`

---

### Category 6: Legal & Contracts (Dates & Money)

**Q11: What is the penalty if a shipping container temperature goes out of range?**
*   **Answer:** If the temperature deviates (outside 2°C-8°C) for more than **2 hours**, the buyer can claim **100% of the cargo value**.
*   **Source:** `vendor_agreement_ocean_freight.md`

**Q12: When will the legacy "SalesTrack v4" system be shut down?**
*   **Answer:** It will be permanently decommissioned on **December 31, 2024**.
*   **Source:** `IT_legacy_retirement_memo.md`