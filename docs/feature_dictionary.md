# Feature Dictionary
* **tenure_years**: tenure / 12 (Human-readable relationship length)
* **avg_monthly_spend**: TotalCharges / max(tenure, 1) (Approximate historical monthly spend)
* **service_count**: count of subscribed services (Engagement/product breadth proxy)
* **is_month_to_month**: Contract == Month-to-month (Known risk segment)
* **auto_pay**: PaymentMethod in automatic categories (Billing behaviour proxy)
* **has_support**: TechSupport == Yes (Support coverage)
* **has_security**: OnlineSecurity == Yes (Security add-on)
* **charge_per_tenure**: MonthlyCharges / max(tenure, 1) (Simple intensity proxy)