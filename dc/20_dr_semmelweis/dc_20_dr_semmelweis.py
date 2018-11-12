## https://projects.datacamp.com/projects/20
## Dr. Semmelweis and the discovery of handwashing

## Task 1
# importing modules
import pandas as pd
import os as os
import matplotlib.pyplot as plt


# Read datasets/yearly_deaths_by_clinic.csv into yearly
fullpathyearly =  os.path.abspath(os.path.join('dc','20_dr_semmelweis','datasets', 'yearly_deaths_by_clinic.csv'))

yearly = pd.read_csv(fullpathyearly)

# Print out yearly
print(yearly)

## Task 2
# Calculate proportion of deaths per no. births
yearly["proportion_deaths"] = yearly["deaths"]/yearly["births"]

# Extract clinic 1 data into yearly1 and clinic 2 data into yearly2
yearly1 = yearly[yearly["clinic"] == "clinic 1"]
yearly2 = yearly[yearly["clinic"] == "clinic 2"]

# Print out yearly1
print(yearly1)

## Task 3
# Plot yearly proportion of deaths at the two clinics
ax = yearly1.plot(x="year", y="proportion_deaths", label="Clinic 1")
yearly2.plot(x="year", y="proportion_deaths", label="Clinic 2", ax = ax)
ax.set_ylabel("Proportion deaths")

# Show the plot
plt.show()

## Task 4
# Read datasets/monthly_deaths.csv into monthly
fullpathmonthly =  os.path.abspath(os.path.join('dc','20_dr_semmelweis','datasets', 'monthly_deaths.csv'))
monthly = pd.read_csv(fullpathmonthly, parse_dates=["date"])

# Calculate proportion of deaths per no. births
monthly["proportion_deaths"] = monthly["deaths"]/monthly["births"]

# Print out the first rows in monthly
print(monthly.head())

## Task 5
# Plot monthly proportion of deaths
ax = monthly.plot(x="date", y="proportion_deaths")
ax.set_ylabel("Proportion deaths")

# Show the plot
plt.show()

## Task 6
# Date when handwashing was made mandatory
handwashing_start = pd.to_datetime('1847-06-01')

# Split monthly into before and after handwashing_start
before_washing = monthly[monthly.date < handwashing_start]
after_washing = monthly[monthly.date >= handwashing_start]

# Plot monthly proportion of deaths before and after handwashing
ax = before_washing.plot(x="date", y="proportion_deaths", label="before")
after_washing.plot(x="date", y="proportion_deaths", label="after", ax = ax)
ax.set_ylabel("Proportion deaths")

# Show the plot
plt.show()

## Task 7
# Difference in mean monthly proportion of deaths due to handwashing
before_proportion = before_washing.proportion_deaths
after_proportion = after_washing.proportion_deaths
mean_diff = after_proportion.mean() - before_proportion.mean()
print(mean_diff)

## Task 8 
# A bootstrap analysis of the reduction of deaths due to handwashing
boot_mean_diff = []
for i in range(3000):
    boot_before = before_proportion.sample(frac=1, replace=True)
    boot_after = after_proportion.sample(frac=1, replace=True)
    boot_mean_diff.append( boot_before.mean() -  boot_after.mean())

# Calculating a 95% confidence interval from boot_mean_diff 
confidence_interval = pd.Series(boot_mean_diff).quantile([0.025, 0.975])
print(confidence_interval)

## Task 9 
# The data Semmelweis collected points to that:
doctors_should_wash_their_hands = True