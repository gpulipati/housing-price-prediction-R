# -----------------------------------------------------------------------------
# Author: Guru Prakash Pulipati
# Batch: Spring 2024
# Problem Statement: Predicting the median house value based on the 
# California Housing data set. 
# -----------------------------------------------------------------------------

library(dplyr)
library(ggplot2)
library(corrplot)
library(reshape2)
library(randomForest)
library(cowplot)

# -----------------------------------------------------------------------------
# 1. Access the Data Set (housing.csv)
# -----------------------------------------------------------------------------

# set the working directory 
# i.e. Session ->  Set Working Directory -> To Source File Location
housing <- read.csv("housing.csv")

dim(housing)
# [1] 20640    10

glimpse(housing, width = 1)
# Rows: 20,640
# Columns: 10
# $ longitude          <dbl> …
# $ latitude           <dbl> …
# $ housing_median_age <dbl> …
# $ total_rooms        <dbl> …
# $ total_bedrooms     <dbl> …
# $ population         <dbl> …
# $ households         <dbl> …
# $ median_income      <dbl> …
# $ median_house_value <dbl> …
# $ ocean_proximity    <chr> …

head(housing,2)
#     longitude latitude housing_median_age total_rooms total_bedrooms population households median_income median_house_value ocean_proximity
# 1   -122.23    37.88                 41         880            129        322        126        8.3252             452600        NEAR BAY
# 2   -122.22    37.86                 21        7099           1106       2401       1138        8.3014             358500        NEAR BAY

housing$ocean_proximity <- as.factor(housing$ocean_proximity)
levels(housing$ocean_proximity)
# [1] "<1H OCEAN"  "INLAND"     "ISLAND"     "NEAR BAY"   "NEAR OCEAN"
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 2. EDA and Data Visualization
# -----------------------------------------------------------------------------

head(housing)
#    longitude latitude housing_median_age total_rooms total_bedrooms population households median_income median_house_value ocean_proximity
# 1   -122.23    37.88                 41         880            129        322        126        8.3252             452600        NEAR BAY
# 2   -122.22    37.86                 21        7099           1106       2401       1138        8.3014             358500        NEAR BAY
# 3   -122.24    37.85                 52        1467            190        496        177        7.2574             352100        NEAR BAY
# 4   -122.25    37.85                 52        1274            235        558        219        5.6431             341300        NEAR BAY
# 5   -122.25    37.85                 52        1627            280        565        259        3.8462             342200        NEAR BAY
# 6   -122.25    37.85                 52         919            213        413        193        4.0368             269700        NEAR BAY

tail(housing)
#        longitude latitude housing_median_age total_rooms total_bedrooms population households median_income median_house_value ocean_proximity
# 20635   -121.56    39.27                 28        2332            395       1041        344        3.7125             116800          INLAND
# 20636   -121.09    39.48                 25        1665            374        845        330        1.5603              78100          INLAND
# 20637   -121.21    39.49                 18         697            150        356        114        2.5568              77100          INLAND
# 20638   -121.22    39.43                 17        2254            485       1007        433        1.7000              92300          INLAND
# 20639   -121.32    39.43                 18        1860            409        741        349        1.8672              84700          INLAND
# 20640   -121.24    39.37                 16        2785            616       1387        530        2.3886              89400          INLAND

summary(housing)
# longitude         latitude     housing_median_age  total_rooms    total_bedrooms  
# Min.   :-124.3   Min.   :32.54   Min.   : 1.00      Min.   :    2   Min.   :   1.0  
# 1st Qu.:-121.8   1st Qu.:33.93   1st Qu.:18.00      1st Qu.: 1448   1st Qu.: 296.0  
# Median :-118.5   Median :34.26   Median :29.00      Median : 2127   Median : 435.0  
# Mean   :-119.6   Mean   :35.63   Mean   :28.64      Mean   : 2636   Mean   : 537.9  
# 3rd Qu.:-118.0   3rd Qu.:37.71   3rd Qu.:37.00      3rd Qu.: 3148   3rd Qu.: 647.0  
# Max.   :-114.3   Max.   :41.95   Max.   :52.00      Max.   :39320   Max.   :6445.0  
# NA's   :207     
#    population      households     median_income     median_house_value   ocean_proximity
#  Min.   :    3   Min.   :   1.0   Min.   : 0.4999   Min.   : 14999     <1H OCEAN :9136  
#  1st Qu.:  787   1st Qu.: 280.0   1st Qu.: 2.5634   1st Qu.:119600     INLAND    :6551  
#  Median : 1166   Median : 409.0   Median : 3.5348   Median :179700     ISLAND    :   5  
#  Mean   : 1425   Mean   : 499.5   Mean   : 3.8707   Mean   :206856     NEAR BAY  :2290  
#  3rd Qu.: 1725   3rd Qu.: 605.0   3rd Qu.: 4.7432   3rd Qu.:264725     NEAR OCEAN:2658  
#  Max.   :35682   Max.   :6082.0   Max.   :15.0001   Max.   :500001  

Req_col <- c("housing_median_age", "total_rooms", "total_bedrooms", "population", 
              "households", "median_income", "median_house_value")

cor_matrix <- cor(housing[,Req_col])
cor_matrix
#                    housing_median_age total_rooms total_bedrooms   population  households median_income median_house_value
# housing_median_age          1.0000000  -0.3612622             NA -0.296244240 -0.30291601  -0.119033990   0.10562341
# total_rooms                -0.3612622   1.0000000             NA  0.857125973  0.91848449   0.198049645   0.13415311
# total_bedrooms                     NA          NA              1           NA          NA            NA           NA
# population                 -0.2962442   0.8571260             NA  1.000000000  0.90722227   0.004834346  -0.02464968
# households                 -0.3029160   0.9184845             NA  0.907222266  1.00000000   0.013033052   0.06584265
# median_income              -0.1190340   0.1980496             NA  0.004834346  0.01303305   1.000000000   0.68807521
# median_house_value          0.1056234   0.1341531             NA -0.024649679  0.06584265   0.688075208   1.00000000

# Strongly positive correlated variables
positive_cor <- sort(cor_matrix[cor_matrix > 0.8 & cor_matrix != 1 & !is.na(cor_matrix)],
                     decreasing = TRUE)
positive_cor
# [1] 0.9184845 0.9184845 0.9072223 0.9072223 0.8571260 0.8571260
# total_rooms & households
# population & household
# total_rooms & population

# Strongly negative correlated variables
negative_cor <- sort(cor_matrix[cor_matrix > -0.5 & cor_matrix <= -1 & !is.na(cor_matrix)],
                     decreasing = TRUE)
negative_cor
# numeric(0)

#------------------------------------------------------------------
# Common plot functions - start
#------------------------------------------------------------------

# Histogram
generate_histogram <- function(data, xaxis, bins = 5) {
  ggplot(data, aes_string(x = xaxis)) +
    geom_histogram(bins = bins, fill = "cornflowerblue", color = "black") +
    labs(title = paste("Histogram of", xaxis), x = xaxis, y = "frequency") +
    geom_vline(xintercept = round(mean(data[[xaxis]], na.rm = TRUE), 2), size = 1, linetype = 4) +
    theme_minimal()
}

# Box plot
generate_boxplot <- function(data, xaxis) {
  # Filter out non-finite values
  filtered_data <- data[is.finite(data[[xaxis]]), ]
  
  # Check if filtered_data is empty
  if (nrow(filtered_data) == 0) {
    print("No finite values found in the specified column.")
    return(NULL)
  }
  
  ggplot(filtered_data, aes_string(y = xaxis)) +
    geom_boxplot(fill = "cornflowerblue", color = "black") +
    labs(title = paste("Boxplot of", xaxis), x = "", y = xaxis) +
    theme_minimal()
}

#Box plot with factor
generate_boxplot_with_factor <- function(data, yaxis, factor_var) {
  ggplot(data, aes_string(x = factor_var, y = yaxis)) +
    geom_boxplot(fill = "cornflowerblue", color = "black") +
    labs(title = paste("Boxplot of", yaxis, "with respect to", factor_var),
         x = factor_var, y = yaxis) +
    theme_minimal()
}

#------------------------------------------------------------------
# Common plot functions - End
#------------------------------------------------------------------
h_plot1 <- generate_histogram(housing, Req_col[1], bins = 20)
h_plot2 <- generate_histogram(housing, Req_col[2], bins = 50)
h_plot3 <- generate_histogram(housing, Req_col[3], bins = 30)
h_plot4 <- generate_histogram(housing, Req_col[4], bins = 20)
h_plot5 <- generate_histogram(housing, Req_col[5], bins = 20)
h_plot6 <- generate_histogram(housing, Req_col[6], bins = 10)
h_plot7 <- generate_histogram(housing, Req_col[7], bins = 10)

plot_grid(h_plot1, h_plot2, h_plot3, h_plot4, h_plot5, h_plot6, 
          h_plot7, ncol = 2)

# Box plots for numeric variable
b_plot1 <- generate_boxplot(housing, Req_col[1])
b_plot2 <- generate_boxplot(housing, Req_col[2])
b_plot3 <- generate_boxplot(housing, Req_col[3])
b_plot4 <- generate_boxplot(housing, Req_col[4])
b_plot5 <- generate_boxplot(housing, Req_col[5])
b_plot6 <- generate_boxplot(housing, Req_col[6])
b_plot7 <- generate_boxplot(housing, Req_col[7])

plot_grid(b_plot1, b_plot2, b_plot3, b_plot4, b_plot5, b_plot6, 
          b_plot7, ncol = 3)

# Box plots for the variables - housing_median_age,
# median_income, and median_house_value “with 
# respect” to the factor variable ocean_proximity.
bf_plot1 <- generate_boxplot_with_factor(housing, Req_col[1], 'ocean_proximity')
bf_plot2 <- generate_boxplot_with_factor(housing, Req_col[6], 'ocean_proximity')
bf_plot3 <- generate_boxplot_with_factor(housing, Req_col[7], 'ocean_proximity')

plot_grid(bf_plot1, bf_plot2, bf_plot3, ncol = 2)

# -----------------------------------------------------------------------------
# 3. DATA TRANSFORMATION
# -----------------------------------------------------------------------------
#
# filling missing values with “statistical median” for total_bedrooms variable
housing$total_bedrooms[is.na(housing$total_bedrooms)] <-
  median(housing$total_bedrooms, na.rm = TRUE)

#checking non_finite_values 
non_finite_values <- housing[!complete.cases(housing[Req_col[3]]), ]
non_finite_values
# [1] longitude          latitude           housing_median_age total_rooms        total_bedrooms    
# [6] population         households         median_income      median_house_value ocean_proximity   
# <0 rows> (or 0-length row.names)

# splitting the ocean_proximity variable into a number of binary categorical
# variables consisting of 1s and 0s. 
categories <- model.matrix(~ocean_proximity - 1, data = housing)
housing_cat <- as.data.frame(categories)
colnames(housing_cat) <- sub("ocean_proximity", "", colnames(housing_cat))

head(housing_cat)
#     <1H OCEAN INLAND ISLAND NEAR BAY NEAR OCEAN
# 1         0      0      0        1          0
# 2         0      0      0        1          0
# 3         0      0      0        1          0
# 4         0      0      0        1          0
# 5         0      0      0        1          0
# 6         0      0      0        1          0

tail(housing_cat)
#         <1H OCEAN INLAND ISLAND NEAR BAY NEAR OCEAN
# 20635         0      1      0        0          0
# 20636         0      1      0        0          0
# 20637         0      1      0        0          0
# 20638         0      1      0        0          0
# 20639         0      1      0        0          0
# 20640         0      1      0        0          0

# joining ocean_proximity binary categorical variables and
# discarding ocean_proximity from housing data frame
housing <- cbind(housing, housing_cat)
housing <- select(housing, -ocean_proximity)

# check exiting column names
colnames(housing)
# [1] "longitude"          "latitude"           "housing_median_age" "total_rooms"        "total_bedrooms"    
# [6] "population"         "households"         "median_income"      "median_house_value" "<1H OCEAN"         
# [11] "INLAND"             "ISLAND"             "NEAR BAY"           "NEAR OCEAN" 

# creating and adding mean_bedrooms & mean_rooms variables using total_bedrooms
# and total_rooms variables as they are more accurate depictions
# of the house
housing$mean_bedrooms = housing$total_bedrooms/housing$households
housing$mean_rooms = housing$total_rooms/housing$households

#discarding total_bedrooms & total_rooms
housing <- select(housing, -total_bedrooms, -total_rooms)

colnames(housing)
# [1] "longitude"          "latitude"           "housing_median_age" "population"         "households"        
# [6] "median_income"      "median_house_value" "<1H OCEAN"          "INLAND"             "ISLAND"            
# [11] "NEAR BAY"           "NEAR OCEAN"         "mean_bedrooms"      "mean_rooms" 

# scaling numerical variable by leaving binary categorical variables  
# and median_house_value which is our response variable
scale_vars <- select(housing, longitude, latitude, housing_median_age, 
                           population, households, median_income, mean_bedrooms,
                           mean_rooms)
housing_scale <- as.data.frame(scale(scale_vars))

head(housing_scale)
# longitude latitude housing_median_age population households median_income mean_bedrooms mean_rooms
# 1 -1.327803 1.052523          0.9821189 -0.9744050 -0.9770092    2.34470896  -0.148510661  0.6285442
# 2 -1.322812 1.043159         -0.6070042  0.8614180  1.6699206    2.33218146  -0.248535936  0.3270334
# 3 -1.332794 1.038478          1.8561366 -0.8207575 -0.8436165    1.78265622  -0.052900657  1.1555925
# 4 -1.337785 1.038478          1.8561366 -0.7660095 -0.7337637    0.93294491  -0.053646030  0.1569623
# 5 -1.337785 1.038478          1.8561366 -0.7598283 -0.6291419   -0.01288068  -0.038194658  0.3447024
# 6 -1.337785 1.038478          1.8561366 -0.8940491 -0.8017678    0.08744452   0.005232996 -0.2697231

# cleaned housing data set after data munging phase 
cleaned_housing <- cbind(
  `NEAR BAY`=housing$`NEAR BAY`, `<1H OCEAN`=housing$`<1H OCEAN`, INLAND=housing$INLAND,
  `NEAR OCEAN`=housing$`NEAR OCEAN`,ISLAND=housing$ISLAND, housing_scale, 
  median_house_value=housing$median_house_value)

# correlation plot on cleaned housing dataframe
# analyzing the correaction for median_house_value shows
# median_income is the strongest among other vars
correlation_matrix <- cor(cleaned_housing, method = "spearman")
col_palette <- colorRampPalette(c("orange", "white", "blue"))(200)
corrplot(correlation_matrix, method = "shade", col = col_palette, tl.cex = 0.8)

# final verification of the data set columns
colnames(cleaned_housing)
# [1] "NEAR BAY"           "<1H OCEAN"          "INLAND"             "NEAR OCEAN"         "ISLAND"            
# [6] "longitude"          "latitude"           "housing_median_age" "population"         "households"        
# [11] "median_income"      "mean_bedrooms"      "mean_rooms"         "median_house_value"

# -----------------------------------------------------------------------------
# 4. TRAINING AND TEST SETS
# -----------------------------------------------------------------------------
# creating 
# random sample index for cleaned_housing data frame, 
# training set with 70% rows 
# test set with rest of the rows 
n <- nrow(cleaned_housing)  # Number of observations
ntrain <- round(n*0.7)      # 70% for training set gives the min test error metric
set.seed(414)               # setting seed for reproducible results

tindex <- sample(n,ntrain)                        # sample index creation
train_cleaned_housing <- cleaned_housing[tindex,] # training set creation
test_cleaned_housing <- cleaned_housing[-tindex,] # test set creation

#validation
nrow(train_cleaned_housing) + nrow(test_cleaned_housing) == nrow(cleaned_housing)
# [1] TRUE

# -----------------------------------------------------------------------------
# 5. SUPERVISED ML - REGRESSION - RANDOMFOREST
# -----------------------------------------------------------------------------
# using randomForest algorithm for training and inference to predict 
# the median house value

#training set 
train_col_index <- which(colnames(train_cleaned_housing) == 'median_house_value')
train_x <- train_cleaned_housing[, -train_col_index]
train_y <- train_cleaned_housing[, train_col_index]

rf = randomForest(x=train_x, y=train_y , ntree=500, importance=TRUE)

names(rf)
# [1] "call"            "type"            "predicted"       "mse"            
# [5] "rsq"             "oob.times"       "importance"      "importanceSD"   
# [9] "localImportance" "proximity"       "ntree"           "mtry"           
# [13] "forest"          "coefs"           "y"               "test"           
# [17] "inbag"  
# -----------------------------------------------------------------------------
# 6. EVALUATING MODEL PERFORMANCE
# -----------------------------------------------------------------------------

# calculating the set root mean squared error (RMSE)
rf
# Call:
#   randomForest(x = train_x, y = train_y, ntree = 500, importance = TRUE) 
# Type of random forest: regression
# Number of trees: 500
# No. of variables tried at each split: 4
# 
# Mean of squared residuals: 2472967347
# % Var explained: 81.52

train_rmse = sqrt(rf$mse[length(rf$mse)])
train_rmse
# [1] 49728.94

#predictions by using the test set
test_col_index <- which(colnames(test_cleaned_housing) == 'median_house_value')
test_x <- test_cleaned_housing[, -test_col_index]
test_y <- test_cleaned_housing[, test_col_index]

y_pred <- predict(rf, test_x)

# test set RMSE
test_rmse = sqrt(mean((test_y - y_pred)^2))
test_rmse
# [1] 48148.52

# the predictions of the median price of a house is within 
# ~$49000.00 of the actual median house price.

# variable importance plot, confirms median_income as the most important feature
varImpPlot(rf)








