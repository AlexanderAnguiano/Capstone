
### Load Libraries ###
library(data.table)
library(zoo)
library(digest)
library(Rcpp)
library(caret)
library(doSNOW)
library(e1071)
library(caretEnsemble)
library(ipred)
library(xgboost)
library(kernlab)
library(elasticnet)
library(lars)
library(MASS)
library(pls)
library(AppliedPredictiveModeling)
library(gtools)
library(stats)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(devtools)
library(gbm)
library(Cubist)
library(party)
library(partykit)
library(randomForest)
library(rpart)
library(RWeka)
library(earth)
library(PerformanceAnalytics)



### Set Working Directory ###
setwd("C:/Users/Your/Desktop")


### Load and Explore Original Training Values and Training Labels ###

original_values <- read.csv("Train_Values.csv", stringsAsFactors = FALSE)
summary(original_values)

original_labels <- read.csv("Train_Labels.csv", stringsAsFactors = FALSE)
summary(original_labels)


### Create Histogram of Prevalence of Undernourishment Training Labels ###
ggplot(data = original_labels, mapping = aes(x = prevalence_of_undernourishment)) + geom_histogram(binwidth = 10)



### Data prepped outside of RStudio using Excel & Talend Data Preparation ###
### Training Labels Prevalence_of_Undernourishment column added to Training Values Table ###
### 7 columns removed from consideration due to number of missing values (see Capstone Report) ###
### Country Code values replaced with numerical "key code" to assist with factorization (see Key) ###
### country Code and Year columns moved to end of Table ###
### Test Values prepared in same fashion as Train Values ###



### Load Prepared Train Data ###
train <- read.csv("prepared_train_values.csv", stringsAsFactors = FALSE)

### Encode Country & Year Varaiables as Factors ###
train$country_code_key <- as.factor(train$country_code_key)
train$year <- as.factor(train$year)



### Impute Missing Train Values using Bagged Decision Trees ###

### Transform all Features to Dummy Variables ###
dummy.vars <- dummyVars(~ ., data = train)
train.dummy <- predict(dummy.vars, train)

### Creat Imputed Data Table ###
pre.process <- preProcess(train.dummy, method = "bagImpute")
imputed.data <- predict(pre.process, train.dummy)


### Use Imputed Data to fill in Missing Values in Training Data ###

train$agricultural_land_area <- imputed.data[, 2]
train$percentage_of_arable_land_equipped_for_irrigation <- imputed.data[, 3]
train$cereal_yield <- imputed.data[, 4]
train$forest_area <- imputed.data[, 5]
train$fertility_rate <- imputed.data[, 7]
train$life_expectancy <- imputed.data[, 8]
train$rural_population <- imputed.data[, 9]
train$population_growth <- imputed.data[, 12]
train$avg_value_of_food_production <- imputed.data[, 13]
train$cereal_import_dependency_ratio <- imputed.data[, 14]
train$food_imports_as_share_of_merch_exports <- imputed.data[, 15]
train$gross_domestic_product_per_capita_ppp <- imputed.data[, 16]
train$imports_of_goods_and_services <- imputed.data[, 17]
train$net_oda_received_percent_gni <- imputed.data[, 18]
train$net_oda_received_per_capita <- imputed.data[, 19]
train$trade_in_services <- imputed.data[, 20]
train$per_capita_food_production_variability <- imputed.data[, 21]
train$per_capita_food_supply_variability <- imputed.data[, 22]
train$avg_supply_of_protein_of_animal_origin <- imputed.data[, 23]
train$caloric_energy_from_cereals_roots_tubers <- imputed.data[, 24]
train$access_to_improved_sanitation <- imputed.data[, 25]
train$access_to_improved_water_sources <- imputed.data[, 26]
train$anemia_prevalence <- imputed.data[, 27]
train$obesity_prevalence <- imputed.data[, 28]
train$open_defecation <- imputed.data[, 29]
train$hiv_incidence <- imputed.data[, 30]
train$access_to_electricity <- imputed.data[, 31]
train$co2_emissions <- imputed.data[, 32]
train$unemployment_rate <- imputed.data[, 33]
train$total_labor_force <- imputed.data[, 34]
train$military_expenditure_share_gdp <- imputed.data[, 35]
train$proportion_of_seats_held_by_women_in_gov <- imputed.data[, 36]
train$political_stability <- imputed.data[, 37]



### Load Prepared Test Data ###
test <- read.csv("prepared_test_values.csv", stringsAsFactors = FALSE)


### Impute Missing Test Values using Bagged Decision Trees ###

### Transform all Features to Dummy Variables ###
test$country_code_key <- as.factor(test$country_code_key)
test$year <- as.factor(test$year)


### Creat Imputed Data Table ###
dummy.varsTEST <- dummyVars(~ ., data = test)
train.dummyTEST <- predict(dummy.varsTEST, test)

### Creat Imputed Data Table ###
pre.processTEST <- preProcess(train.dummyTEST, method = "bagImpute")
imputed.dataTEST <- predict(pre.processTEST, train.dummyTEST)

### Use Imputed Data to fill in Missing Values in Test Data ###

test$percentage_of_arable_land_equipped_for_irrigation <- imputed.dataTEST[, 2]
test$cereal_yield <- imputed.dataTEST[, 3]
test$avg_value_of_food_production <- imputed.dataTEST[, 12]
test$cereal_import_dependency_ratio <- imputed.dataTEST[, 13]
test$food_imports_as_share_of_merch_exports <- imputed.dataTEST[, 14]
test$imports_of_goods_and_services <- imputed.dataTEST[, 16]
test$net_oda_received_percent_gni <- imputed.dataTEST[, 17]
test$net_oda_received_per_capita <- imputed.dataTEST[, 18]
test$trade_in_services <- imputed.dataTEST[, 19]
test$per_capita_food_production_variability <- imputed.dataTEST[, 20]
test$per_capita_food_supply_variability <- imputed.dataTEST[, 21]
test$avg_supply_of_protein_of_animal_origin <- imputed.dataTEST[, 22]
test$caloric_energy_from_cereals_roots_tubers <- imputed.dataTEST[, 23]
test$access_to_improved_water_sources <- imputed.dataTEST[, 25]
test$anemia_prevalence <- imputed.dataTEST[, 26]
test$obesity_prevalence <- imputed.dataTEST[, 27]
test$open_defecation <- imputed.dataTEST[, 28]
test$hiv_incidence <- imputed.dataTEST[, 29]
test$access_to_electricity <- imputed.dataTEST[, 30]
test$co2_emissions <- imputed.dataTEST[, 31]
test$unemployment_rate <- imputed.dataTEST[, 32]
test$total_labor_force <- imputed.dataTEST[, 33]
test$military_expenditure_share_gdp <- imputed.dataTEST[, 34]
test$proportion_of_seats_held_by_women_in_gov <- imputed.dataTEST[, 35]
test$political_stability <- imputed.dataTEST[, 36]


### Since predictions will be made against new data, the uniuqe identifier country_code_key will be removed from Train & Test data. ###
train <- subset(train, select = -country_code_key)
test <- subset(test, select = -country_code_key)

### View the structure of the data frames and encoding. ###
str(train)
str(test)



### Tuning parameters are set for 10-fold cross validation repeated 5 times. ###
train.control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, search = "grid")


### All models will run 6 clusters of the reapeated cross validation (Caution - CPU intensive, reduce clusters if necessary). ###
### Set all seeds to same number so that resamples of hold out sets can be tested against each other. ###


### Linear Regression Model ###
cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

set.seed(100)
linReg <- train(prevalence_of_undernourishment ~ ., 
                data = train, 
                method = "lm",
                trControl = train.control)
stopCluster(cl)


### Partial Least Squares Model ###
cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

set.seed(100)
plsModel <- train(prevalence_of_undernourishment ~ ., 
                  data = train, 
                  method = "pls", 
                  preProcess = c("center", "scale"), 
                  tuneLength = 15, 
                  trControl = train.control)
stopCluster(cl)


### Penalized Regression Model (elasticnet lasso) ###
enetGrid <- expand.grid(.lambda = c(0, .001, 0.01, .1),
                        .fraction = seq(.05, 1, length = 20))

cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

set.seed(100)
enetTune <- train(prevalence_of_undernourishment ~ ., data = train,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = train.control,
                  preProcess = c("center", "scale"))
stopCluster(cl)


### Multivariate Adaptive Regression Splines (MARS) ###
cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

set.seed(100)
earthModel <- train(prevalence_of_undernourishment ~ ., 
                    data = train, 
                    method = "earth", 
                    tuneGrid = expand.grid(.degree = 1, 
                                           .nprune = 2:25), 
                    trControl = train.control)
stopCluster(cl)


### Support Vector Machine ###
cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

set.seed(100)
svmR.cv <- train(prevalence_of_undernourishment ~ ., 
                 data = train, 
                 method = "svmRadial", 
                 preProcess = c("center", "scale"), 
                 tuneLength = 15, trControl = train.control)               
stopCluster(cl)


### Boosted Tree ###
gbm.grid <- expand.grid(.interaction.depth = seq(1, 7, by = 2), 
                        .n.trees = seq(100, 1000, by = 50), 
                        .n.minobsinnode = 10, 
                        .shrinkage = c(0.01, 0.1))

cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

set.seed(100)
gbmTune <- train(prevalence_of_undernourishment ~ ., 
                 data = train, 
                 method = "gbm", 
                 tuneGrid = gbm.grid, 
                 verbose = FALSE, 
                 trControl = train.control)
stopCluster(cl)


### Cubist ###
cubistGrid <- expand.grid(.committees = c(1, 5, 10, 50, 75, 100), 
                          .neighbors = c(0, 1, 3, 5, 7, 9))

cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

set.seed(100)
cbModel <- train(prevalence_of_undernourishment ~ ., 
                 data = train, 
                 method = "cubist", 
                 tuneGrid = cubistGrid, 
                 trControl = train.control)
stopCluster(cl)



### RMSE & R2 plots of resampled hold out sets. ###

resamples2 <- resamples(list("Linear Reg" = linReg,
                             "MARS" = earthModel,
                             "Lasso Net" = enetTune,
                             "PLS" = plsModel,
                             "SVM" = svmR.cv, 
                             "Boosted Tree" = gbmTune, 
                             "Cube" = cbModel))


parallelplot(resamples2, metric = "RMSE")
parallelplot(resamples2, metric = "Rsquared")


### Predictions against New Data ###

pred.linReg <- predict(linReg, test)
pred.plsModel <- predict(plsModel, test)
pred.enet <- predict(enetTune, test)
pred.MARS <- predict(earthModel, test)
pred.svm <- predict(svmR.cv, test)
pred.gbm <- predict(gbmTune, test)
pred.cube <- predict(cbModel, test)


### All predictions can be combined into one data drame and exported to an Excel csv. ###
Predictions <- data.frame(pred.linReg, pred.plsModel, pred.enet, pred.MARS, pred.svm, pred.gbm, pred.cube)

col_headings <- c('linReg', 'pls', 'enet', 'MARS', 'svm', 'gbm', 'Cube')
colnames(Predictions) <- col_headings

write.csv(Predictions, file = "AllCapstonePredictions.csv")


### RMSE plot indicates the Cubist Model performed the best. Evaluate and export separately. ###

### Variable Importance ###
varImp(cbModel)
top.twenty <- varImp(cbModel)
plot(top.twenty, top = 20)

### Create Correlation Data Frame of Top 3 Important Variables from Original Data ###
Correlation <- data.frame(original_labels$prevalence_of_undernourishment,
                          original_values$urban_population,
                          original_values$access_to_improved_water_sources,
                          original_values$obesity_prevalence)

### Abbreviate Column Names for Better Fit and View in Correlation Matrix ###
corr_headings <- c('Prev.Undrnrshnt.', 'Urb.Pop.', 'Acc.Imp.H2O', 'ObesityPrev.')
colnames(Correlation) <- corr_headings

### Correlation Matrix ###
chart.Correlation(Correlation, histogram = TRUE, pch = 19)

### Export Cubist Predictions Separately ###
Cubist <- data.frame(pred.cube)
cubist_headings <- c('Cubist')
colnames(Cubist) <- cubist_headings
write.csv(Cubist, file = "CapstoneCubistPrediction.csv")







