# Incident Post-Mortem: INC-402 (Saarthi Cloud Outage)
**Date:** August 14, 2023
**Severity:** High (Sev-1)
**Affected Service:** Saarthi "Cloud-Bridge" API Gateway (EU-West Region)
**Duration:** 42 Minutes

## Summary
At 14:05 UTC, the API Gateway handling traffic for our FinTech clients (specifically FinSecure Group) experienced a 100% packet drop. This resulted in failed transaction requests for approximately 12,000 end-users.

## Root Cause Analysis (RCA)
*   **Primary Cause:** A misconfigured Terraform script applied during the daily 14:00 deployment window.
*   **Specific Error:** The script inadvertently modified the `max_connections` parameter on the Load Balancer from `10,000` to `0`.
*   **Trigger:** Deployment ID #D-9982 initiated by Junior DevOps Engineer (Manual override was used).

## Resolution
*   **14:10:** Automated alerts triggered in the SOC.
*   **14:15:** Rollback script failed due to database lock.
*   **14:35:** Rajesh Kumar (Principal Architect) manually flushed the connection pool and restored the previous configuration snapshot.
*   **14:47:** Traffic normalized.

## Action Items
1.  **Immediate:** Disable manual overrides for Junior Engineers on Production environments (Assigned to: Maya Singh).
2.  **Process:** Implement "Canary Deployments" for the EU-West region to catch config errors before full rollout.
3.  **Audit:** Review all Terraform state files for the FinSecure project.