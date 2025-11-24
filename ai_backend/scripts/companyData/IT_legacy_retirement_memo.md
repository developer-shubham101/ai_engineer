# Technical Memo: Sunset of "Legacy-Agni" CRM
**To:** Sales & IT Departments
**From:** Group IT Architecture Team

## Announcement
Effective **December 31, 2024**, the legacy on-premise CRM system ("SalesTrack v4", also known as the "Blue Screen") will be permanently decommissioned.

## The New Standard
All sales data must be migrated to **Salesforce Cloud** by October 30, 2024.

## Why are we doing this?
1.  **Security:** SalesTrack v4 runs on Windows Server 2012, which is no longer supported by Microsoft.
2.  **Integration:** The old system cannot connect to VajraScope for analytics.
3.  **Cost:** Maintaining the physical servers for SalesTrack costs $200k/year.

## Migration Schedule
*   **June 2024:** Data cleaning (Sales teams to delete duplicates).
*   **August 2024:** Read-only mode enabled on SalesTrack.
*   **October 2024:** Final data sync.
*   **December 31, 2024:** Server shutdown.

**Warning:** Any data not migrated by the deadline will be archived to cold storage tape (Glacier) and will require a written request to retrieve, taking up to 7 days.