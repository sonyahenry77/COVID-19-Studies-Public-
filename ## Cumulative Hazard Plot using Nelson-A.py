## Cumulative Hazard Plot using Nelson-Aalen Estimator

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import NelsonAalenFitter
from lifelines.plotting import add_at_risk_counts

# Load  data
data = pd.read_csv("matched_long_anderson_gill_hosp_2026.csv")

# Extra security check to ensure tstop is within 0 to 60 months
data = data[(data['tstop'] > 0) & (data['tstop'] <= 60)]

#  NelsonAalenFitter
naf_non_covid = NelsonAalenFitter()
naf_hospitalized_covid = NelsonAalenFitter()
#naf_not_hospitalized_covid = NelsonAalenFitter()

#  categories
non_covid_df = data[data['category'] == 'non_covid']
hospitalized_covid_df = data[data['category'] == 'covid_hosp']
#not_hospitalized_covid_df = data[data['category'] == 'covid_nonhosp']

# Fit model 
naf_non_covid.fit(durations=non_covid_df['tstop'], event_observed=non_covid_df['event'], label='Non-COVID')
naf_hospitalized_covid.fit(durations=hospitalized_covid_df['tstop'], event_observed=hospitalized_covid_df['event'], label='Hospitalized due to COVID')
#naf_not_hospitalized_covid.fit(durations=not_hospitalized_covid_df['tstop'], event_observed=not_hospitalized_covid_df['event'], label='Non-hospitalized COVID +ve')

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
naf_non_covid.plot_cumulative_hazard(ax=ax, linewidth=2, ci_show=True, color='green', linestyle='--')
naf_hospitalized_covid.plot_cumulative_hazard(ax=ax, linewidth=2, ci_show=True, color='red')
#naf_not_hospitalized_covid.plot_cumulative_hazard(ax=ax, linewidth=2, ci_show=True, color='blue')


ax.set_xticks(range(0, 60, 5))  # Add x-ticks every 6 months for more granularity

# Add at-risk counts 
#add_at_risk_counts(naf_non_covid, naf_hospitalized_covid, naf_not_hospitalized_covid, ax=ax, fontsize=16)

ax.set_xlim(left=0, right=55)  # Limit the x-axis to 36 months
ax.set_ylim(bottom=0, top=1.4)  # Start y-axis at 0
ax.set_xlabel('Time (months)', fontsize=14)
ax.set_ylabel('Cumulative Hazard (Exacerbations)', fontsize=14)
ax.legend(fontsize=12)
ax.tick_params(axis='both', labelsize=12)
plt.title('                                                                                                   ', fontsize=16)
ax.grid(False)

output_path = "cumulative_hazard_plot_hosp.png"  # Change the file name or path as needed
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.show()


