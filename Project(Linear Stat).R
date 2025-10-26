
library(ggplot2)
require(reshape2)
require(lme4)
library(effects)
library(tidyverse)
library(caret)
library(car)
library(ResourceSelection)
library(pROC)
library(pscl)
library(survey)

#import dataset
data <- read.csv(file.choose())


#Verifying my dependent variable
table(data$admit)

#Hence, the dependent variable is a binary

# Use Variance Inflation Factor (VIF) to detect multicollinearity
model <- glm(admit ~ gre + gpa + rank, data = data, family = binomial)
summary(model)

# Extract coefficients
coef_summary <- summary(model)$coefficients

# Calculate Odds Ratios (OR) and 95% Confidence Intervals (CI)
odds_ratios <- exp(coef_summary[, "Estimate"])
conf_int <- exp(confint(model))  # 95% confidence intervals
p_values <- coef_summary[, "Pr(>|z|)"]

# Combine results into a data frame
results <- data.frame(
  Estimate = coef_summary[, "Estimate"],
  Odds_Ratio = odds_ratios,
  `2.5% CI` = conf_int[, 1],
  `97.5% CI` = conf_int[, 2],
  P_Value = p_values
)

# Print the results
print(results)

model_stepwise <- glm(admit ~ gre+gpa+rank+gre*gpa+gre*rank+gpa*rank, data = data, family = binomial)
null=glm(admit ~ 1, data = data, family = binomial)
step(null,scope=list(lower=null,upper=model_stepwise),direction="both")

model_2 <- glm(admit ~  rank + gpa + gre + gpa:gre, data = data, family = binomial)
summary(model_2)


#Goodness of Fit

anova(model,test="Chisq")

cooks.distance<-cooks.distance(model)
which(cooks.distance>1)


#The model should fit the data adequately. Using Hosmer-Lemeshow Test


# Hosmer-Lemeshow test
hoslem.test(data$admit, fitted(model),g=10)

#Wald Test to determine if predictors are significant
regTermTest(model,"gpa")
regTermTest(model,"gre")
regTermTest(model,"rank")

# Check correlation for numeric variables
Iv=data[c("gre", "gpa", "rank")]
cor(Iv)

vif(model)

# Add interaction terms for log-transformed predictors
data$log_gre <- log(data$gre)
data$log_gpa <- log(data$gpa)
data$log_rank <- log(data$rank)
model_lin <- glm(admit ~ gre*log_gre + gpa*log_gpa + rank*log_rank, data = data, family = binomial)
summary(model_lin)

pseudo_r2 <- pR2(model)
print(pseudo_r2)

plot(allEffects(model))

#Area Under the Curve (AUC).
#Sensitivity and specificity.


# ROC Curve and AUC
roc_obj <- roc(data$admit, fitted(model))
auc(roc_obj)
plot(roc_obj, main = "ROC Curve", col = "blue", lwd = 2)

#Training and Testing Model


# Make the dependent variable binary (factor)
data$admit <- as.factor(data$admit)

# Split into training (80%) and testing (20%) sets
set.seed(123)  # For reproducibility
train_index <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

#Fit the model using the training data.

# Train the logistic regression model
log_model <- glm(admit ~ gre + gpa + rank, data = train_data, family = binomial)

# Summary of the model
#summary(log_model)

# Predict probabilities on the test set
test_data$pred_prob <- predict(log_model, newdata = test_data, type = "response")

# Convert probabilities to binary predictions (threshold = 0.5)
test_data$pred_class <- ifelse(test_data$pred_prob > 0.5, 1, 0)


# Create confusion matrix
conf_matrix <- confusionMatrix(as.factor(test_data$pred_class), test_data$admit)

# Print the confusion matrix
print(conf_matrix)

# Calculate accuracy manually
accuracy <- mean(test_data$pred_class == test_data$admit)
print(paste("Accuracy:", round(accuracy, 2)))

varImp(model)

