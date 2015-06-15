#####################
### Load the data ###
#####################

library(dplyr)
library(lubridate)
library(ggplot2)
library(stringr)
library(gridExtra)
library(tidyr)
library(Hmisc)
library(caret)

setwd("~/Documents/SF Crime")
train <- read.csv('train.csv')
test <- read.csv('test.csv')
sampleSubmission <- read.csv('sampleSubmission.csv')



#######################
### Data processing ###
#######################

train_df <- tbl_df(train)
test_df <- tbl_df(test)
sampleSubmission_df <- tbl_df(sampleSubmission)


# "Dates" variable.
train$Dates <- ymd_hms(train$Dates, tz = 'GMT')
test$Dates <- ymd_hms(test$Dates, tz = 'GMT')
train_df$Dates <- floor_date(train$Dates,'month')
test_df$Dates <- floor_date(test$Dates,'month')



################################
### Classification algorithm ###
################################

# We decide to try to identify each crime based on the variables
# "Dates", "PdDistrict", "Category". For each crime observed in 
# a specific district at a specific date, we calcluate the probability
# for this crime to be of such or such category based on the data
# over the entire month during during which the crime occured.

# It is therefore natural to aggregate the training dataset "train_df"
# and use join operations between datasets to obtain the solutions.


## let's build our prediction algorithm.

classification_algo_1 <- function(training_data, testing_data){
        
        # data set showing the probability of a crime category in a given 
        # district at a given month
        probas_df <- training_data %>%
                group_by(Dates, PdDistrict, Category) %>%
                summarise(Count = n()) %>%
                mutate(Total = sum(Count), Probas = round(Count/Total,3)) %>%   
                select(Dates, PdDistrict, Category, Probas) 
        
        
        # prediction dataset
        testing_data <- testing_data %>%
                mutate(Id = 0:(dim(testing_data)[1]-1))
        
        prediction_df <- testing_data %>%
                select(Id, Dates, PdDistrict) %>%
                left_join( . , probas_df, by = c('Dates', 'PdDistrict')) %>%
                select(Id, Category, Probas) %>% 
                spread( . , Category, Probas) %>%
                as.matrix

        prediction_df[is.na(prediction_df)] <- 0
        prediction_df <- tbl_df(as.data.frame(prediction_df))
        prediction_df
} 

classification_algo_1(train_df, test_df)


###################################
### Logarithmic Loss evaluation ###
###################################

# Logarithmic Loss function
MultiLogLoss <- function(act, pred){
        eps = 1e-15;
        nr <- nrow(pred)
        pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr)      
        pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
        ll = sum(act*log(pred) )
        ll = ll * -1/(nrow(act))      
        return(ll);
}


# Cross-validation...
set.seed(125)
inTrain <- createDataPartition(train_df$Category, 
                               p = 0.7, list = F, times = 5)

evaluations_1 <- NULL
for (j in 1:5){
        training_df <- slice(train_df, inTrain[,j])
        
        testing_df <- slice(train_df, -inTrain[,j])
        testing_df <- testing_df %>%
                mutate(Id = 0:(dim(testing_df)[1]-1))
        
        actual_df <- testing_df %>%
                select(Id, Category) %>%
                mutate(Probas = 1) %>%
                spread( . , Category, Probas) %>%
                as.matrix
        actual_df[is.na(actual_df)] <- 0
        actual_df <- tbl_df(as.data.frame(actual_df))
        
        prediction_df <- classification_algo_1(training_df, testing_df)
        evaluations_1[j] <- MultiLogLoss(actual_df[-1], prediction_df[-1])
}

evaluations_1

# 3.141433 3.170781 3.134323 3.168588 3.162994


###################
### Submission! ###
###################

submission_1_df <- classification_algo_1(train_df, test_df)
names(submission_1_df) <- names(sampleSubmission_df)

names <- names(sampleSubmission_df)
names <- str_replace_all(string = names, pattern = '[\\.]', replacement = ' ')
names[30:31] <- c("SEX OFFENSES FORCIBLE", "SEX OFFENSES NON FORCIBLE")
names(submission_1_df) <- names

write.csv(submission_1_df, "submission_1.csv", quote=F, row.names=F)


