"""Generate a synthetic but realistic telco customer-churn dataset.

The generator builds churn as a (noisy) logistic function of a few drivers so
that models trained on the data find genuine, interpretable signal:

  * month-to-month contracts, high monthly charges and short tenure -> churn
  * long tenure, two-year contracts and auto-pay -> retention

Run:  python data/generate_data.py
Output: data/churn.csv  (3000 rows)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 3000


def main() -> None:
    contract = RNG.choice(
        ["Month-to-month", "One year", "Two year"],
        size=N,
        p=[0.55, 0.25, 0.20],
    )
    internet = RNG.choice(
        ["Fiber optic", "DSL", "No"], size=N, p=[0.44, 0.34, 0.22]
    )
    payment = RNG.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=N,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    tenure = RNG.integers(0, 73, size=N)                      # months
    monthly_charges = np.round(RNG.normal(70, 30, size=N).clip(18, 130), 2)
    total_charges = np.round(monthly_charges * tenure + RNG.normal(0, 25, N), 2)
    total_charges = total_charges.clip(0, None)

    senior = RNG.choice([0, 1], size=N, p=[0.84, 0.16])
    partner = RNG.choice([0, 1], size=N, p=[0.52, 0.48])
    paperless = RNG.choice([0, 1], size=N, p=[0.41, 0.59])
    support_calls = RNG.poisson(1.2, size=N)

    # ---- latent churn propensity (log-odds) ----
    logit = (
        -1.2
        + 1.4 * (contract == "Month-to-month")
        - 1.1 * (contract == "Two year")
        + 0.9 * (internet == "Fiber optic")
        + 0.8 * (payment == "Electronic check")
        - 0.035 * tenure
        + 0.015 * (monthly_charges - 70)
        + 0.25 * support_calls
        + 0.4 * senior
        - 0.3 * partner
        + 0.2 * paperless
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (RNG.random(N) < prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(N)],
            "senior_citizen": senior,
            "partner": np.where(partner == 1, "Yes", "No"),
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "paperless_billing": np.where(paperless == 1, "Yes", "No"),
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_calls": support_calls,
            "churn": np.where(churn == 1, "Yes", "No"),
        }
    )

    # inject a few realistic missing values in total_charges (new customers)
    new_mask = df["tenure_months"] == 0
    df.loc[new_mask, "total_charges"] = np.nan

    out = "data/churn.csv"
    df.to_csv(out, index=False)
    rate = (df["churn"] == "Yes").mean()
    print(f"Wrote {out}: {len(df)} rows, churn rate = {rate:.1%}")


if __name__ == "__main__":
    main()
