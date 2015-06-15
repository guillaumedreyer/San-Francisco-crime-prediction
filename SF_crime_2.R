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


# "X_interval", "Y_interval" and "Location" variables
X_cut <- cut2(train_df$X, g = 5, digits = 7, onlycuts = T)
Y_cut <- cut2(train_df$Y, g = 5, digits = 7, onlycuts = T)

train_df <- train_df %>% mutate(X_interval = cut(X, breaks = X_cut, 
                                                 dig.lab = 7, right = F, 
                                                 include.lowest = T),
                                Y_interval = cut(Y, breaks = Y_cut, 
                                                 dig.lab = 7, right = F, 
                                                 include.lowest = T),
                                Location = paste(X_interval,'*', Y_interval)) 

test_df <- test_df %>% mutate(X_interval = cut(X, breaks = X_cut, 
                                               dig.lab = 7, right = F, 
                                               include.lowest = T),
                              Y_interval = cut(Y, breaks = Y_cut, 
                                               dig.lab = 7, right = F, 
                                               include.lowest = T),
                              Location = paste(X_interval,'*', Y_interval))

sapply(train_df, function(x){sum(is.na(x))})
sapply(test_df, function(x){sum(is.na(x))})



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

classification_algo_2 <- function(training_data, testing_data){
        
        # data set showing the probability of a crime category in a given 
        # location at a given month
        probas_df <- group_by(training_data, 
                              Dates,   
                              Category,
                              Location) %>%
                summarise(Count = n()) %>%
                mutate(Total = sum(Count), Probas = round(Count/Total,3)) %>%   
                select(Dates, Category, Location, Probas) 
        
        
        # prediction dataset
        testing_data <- testing_data %>%
                mutate(Id = 0:(dim(testing_data)[1]-1))
        
        prediction_df <- testing_data %>%
                select(Id, Dates, Location) %>%
                left_join( . , probas_df, by = c('Dates', 'Location')) %>%
                select(Id, Category, Probas) %>% 
                spread( . , Category, Probas) %>%
                as.matrix
        
        prediction_df[is.na(prediction_df)] <- 0
        prediction_df <- tbl_df(as.data.frame(prediction_df))
        prediction_df
}

classification_algo_2(train_df, test_df)


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

evaluations_2 <- NULL
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
        
        prediction_df <- classification_algo_2(training_df, testing_df)
        evaluations_2[j] <- MultiLogLoss(actual_df[-1], prediction_df[-1])
}

evaluations_2

# 0.7523271 0.8002612 0.7767444 0.7651924 0.7838247




###################
### Submission! ###
###################

submission_2_df <- classification_algo_2(train_df, test_df)
names(submission_2_df) <- names(sampleSubmission_df)

names <- names(sampleSubmission_df)
names <- str_replace_all(string = names, pattern = '[\\.]', replacement = ' ')
names[30:31] <- c("SEX OFFENSES FORCIBLE", "SEX OFFENSES NON FORCIBLE")
names(submission_2_df) <- names

write.csv(submission_2_df, "submission_2.csv", quote=F, row.names=F)
       



