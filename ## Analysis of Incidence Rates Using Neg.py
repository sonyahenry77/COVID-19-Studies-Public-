## Analysis of Incidence Rates Using Negative Binomial Regression


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv("matched_long_anderson_gill_nonhosp_2026.csv")

# Compute follow-up time (in years) 
df['person_time_years'] = (df['tstop'] - df['tstart']) / 12

# ------------------------------------------------------------------
person_df = (
    df.groupby([
       'PERSON_ID', 'category','encounter_1year', 'vaccinated', 'frequent_exacerbator_pre'
    ], as_index=False)
    .agg(
        total_events=('event', 'sum'),
        total_person_time_years=('person_time_years', 'sum')
    )
)

# Extra security; Keep only those with >0 follow-up time
person_df = person_df[person_df['total_person_time_years'] > 0].copy()

# ------------------------------------------------------------------
person_df['category'] = person_df['category'].replace({'non_covid': 'Non_COVID'})
person_df = person_df[person_df['category'].isin(['covid_nonhosp', 'Non_COVID'])].copy()
person_df['category'] = pd.Categorical(person_df['category'], categories=['Non_COVID', 'covid_nonhosp'])

# Offset for log(person-time)
person_df['log_person_time'] = np.log(person_df['total_person_time_years'])

# ------------------------------------------------------------------
crude_rates = (
    person_df.groupby('category')
    .agg(
        total_events=('total_events', 'sum'),
        total_person_time=('total_person_time_years', 'sum')
    )
    .assign(
        crude_rate_per_100py=lambda x: (x['total_events'] / x['total_person_time']) * 100
    )
    .reset_index()
)

print("\n Crude Incidence Rates (per 100 PY):")
print(crude_rates.round(2))

# ---Fit Negative Binomial regression model
model_nb = smf.glm(
    formula=(
        "total_events ~ C(category) + vaccinated + encounter_1year+ frequent_exacerbator_pre"
    ),
    data=person_df,
    family=sm.families.NegativeBinomial(),
    offset=person_df['log_person_time']
).fit(cov_type='HC0')  # Robust SEs

print("\n Negative Binomial Model Summary:")
print(model_nb.summary())

# ------------------------------------------------------------------
params = model_nb.params
conf = model_nb.conf_int()
conf.columns = ["2.5%", "97.5%"]

results_df = pd.DataFrame({
    "Covariate": params.index,
    "IRR": np.exp(params),
    "Lower 95% CI": np.exp(conf["2.5%"]),
    "Upper 95% CI": np.exp(conf["97.5%"]),
    "P-value": model_nb.pvalues
}).round(3)

print("\n Adjusted Incidence Rate Ratios (IRRs) — Negative Binomial:")
print(results_df)

# ------------------------------------------------------------------
base_rate = np.exp(model_nb.params['Intercept']) * 100  # per 100 PY

adj_rates = {
    'Non_COVID': base_rate,
    'covid_nonhosp': base_rate * np.exp(model_nb.params.get('C(category)[T.covid_nonhosp]', 0))
}

adj_rates_df = pd.DataFrame(list(adj_rates.items()), columns=['category', 'adjusted_rate_per_100py'])
adj_rates_df['adjusted_rate_per_100py'] = adj_rates_df['adjusted_rate_per_100py'].round(2)

print("\n Model-based Adjusted Incidence Rates (per 100 PY) — Negative Binomial:")
print(adj_rates_df)

# ------------------------------------------------------------------
final_rates = crude_rates.merge(adj_rates_df, on='category', how='outer')
print("\n Combined Crude vs Adjusted Rates (per 100 PY):")
print(final_rates.round(2))

# ------------------------------------------------------------------
# Fit Poisson model for comparison
poisson_model = smf.glm(
    formula=model_nb.model.formula,
    data=person_df,
    family=sm.families.Poisson(),
    offset=person_df['log_person_time']
).fit(cov_type='HC0')

print("\n Model fit comparison:")
print(f"Poisson AIC: {poisson_model.aic:.1f}")
print(f"NegBin  AIC: {model_nb.aic:.1f}")

if model_nb.aic < poisson_model.aic:
    print(" Negative Binomial provides a better fit (handles overdispersion).")
else:
    print(" Poisson may be adequate (no strong overdispersion evidence).")
