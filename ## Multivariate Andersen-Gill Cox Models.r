## Multivariate Andersen-Gill Cox Models with IPW Weights in R( Modify as needed with IPW weights and covariates)

library(survival)
library(knitr)
library(car)


data <- read.csv("CSVfile.csv") 

# weights are numeric
data$ipw_weight_clipped <- as.numeric(data$ipw_weight_clipped)

#  reference levels
data$INSURANCE <- as.factor(data$INSURANCE)
data$category <- as.factor(data$category)

data$INSURANCE <- relevel(data$INSURANCE, ref = "Private")
data$category <- relevel(data$category, ref = "Non_COVID")

# race variable
data$race <- as.factor(ifelse(data$black_non == 1, "Black",
                              ifelse(data$Hisp == 1, "Hispanic",
                                     ifelse(data$other_race == 1, "Other", "White"))))
data$race <- relevel(data$race, ref = "White")

# insurance variable
data$Significant_INSURANCE <- ifelse(data$INSURANCE == "Medicaid", "Medicaid", 
                                     ifelse(data$INSURANCE == "Medicare", "Medicare", "Private"))
data$Significant_INSURANCE <- factor(data$Significant_INSURANCE, levels = c("Private", "Medicaid", "Medicare"))



# Create survival object
surv_obj <- Surv(time = data$tstart, time2 = data$tstop, event = data$event, type = 'counting')


#covariates <- c("category"#)


#covariates <- c("category","AGE", "Gndrbool", "obesity", "smoking","race", "Significant_INSURANCE",
#                "hypertension", "diabetes1", "diabetes2", "CKD", "liver_disease", "CHF", "CAD", "MI",
#                "Asthma", "Osa", "pulhp", "frequent_exacerbator_pre","encounter_1year", "vaccinated") 


#covariates <- c( "frequent_exacerbator_pre","category", "vaccinated", "encounter_1year")


# formula
formula <- as.formula(paste("surv_obj ~", paste(covariates, collapse = " + "), "+ cluster(PERSON_ID)"))

# Fit weighted Cox model using IPW
multivariate_model <- coxph(formula, data = data, weights = ipw_weight_clipped, method = "breslow", robust = TRUE)

# Alternative: Without IPW weights
#multivariate_model <- coxph(formula, data = data, method = "breslow", robust = TRUE)

# Model summary
summary(multivariate_model)

# Extract HRs and CIs
coefficients <- summary(multivariate_model)$coefficients
ci <- confint(multivariate_model)

HR <- exp(coefficients[, "coef"])
P.Value <- coefficients[, "Pr(>|z|)"]
LowerCI <- exp(ci[, 1])
UpperCI <- exp(ci[, 2])

multivariate_results <- data.frame(
  Covariate = rownames(coefficients),
  HR = HR,
  LowerCI = LowerCI,
  UpperCI = UpperCI,
  P.Value = P.Value,
  stringsAsFactors = FALSE
)

# Output results
knitr::kable(multivariate_results, format = "pipe", caption = "Results of Multivariate Andersen-Gill Cox Model (IPW)",
             col.names = c("Covariate", "HR", "95% Lower", "95% Upper", "P-value"))



