
library(dplyr)
library(nnet)
library(caret)
library(e1071)
library(randomForest)
library(party) # for cfrorest
library(rowr)
library(DataCombine)

## Read the data
data <- read.csv("data/dataset.txt", header = FALSE, stringsAsFactors=FALSE)

## small data for test
#data <- read.csv("data/test_data.txt", header = FALSE, stringsAsFactors=FALSE)

## Assign the column names
colnames(data) <- c("user","activity","timestamp","xaccel","yaccel","zaccel")
#colnames(test.data) <- c("user","activity","timestamp","xaccel","yaccel","zaccel")

## Drop the last column that is wrongly interpreted as column by R
## because each line ends in ","
data <- data[, -7]
#test.data <- test.data[, -7]

## Order the data by user
#order.data <- arrange(data, activity)
# 
# ## Create a new column with the difference of timestamp between one observation
# ## and the previous one
# data <-mutate(data, tdiff = timestamp - lag(timestamp))
# #test.data <-mutate(test.data, tdiff = timestamp - lag(timestamp))
# 
# ## Check the difference of timestamps > 50 ms
# filter(data, tdiff > 50)
# 
# ## Check some values
# filter(data, user == 1793 & activity == "Stairs" & tdiff > 0)
# 
# ## Let's group the observations using a window of 100
# filter(order.data, activity == "Walking")
# 
# # Find when a user changed
# filter(order.data, user != lag(user))
# filter(data, user != lag(user))
# # Find when an activity changed
# filter(order.data, activity != lag(activity))
# filter(data, activity != lag(activity))

######################################################################
## Check which are the indices when user changes
#user.indexes <- which(data$user != lag(data$user))
act.indexes <- which(data$activity != lag(data$activity))
## Add the firt onbservation index
act.indexes <- c(1,  act.indexes)

## Create the windowed data frame.
w.data <-data.frame()
act <- data.frame()

w.size = 100 ## window size
end = nrow(data) ## end of file

## Create the new vectors using w.size observations
pb <- txtProgressBar(min = 0, max = end, style = 3) ## progress bar

## loop for activity
for (ac in act.indexes){
        #print(c("ac",ac))
        
        ## loop for window
        for (i in seq(ac, end, w.size)) {
                #print(c("i", i))
                #print(c("first user", data[(i + w.size-1),]$user)) 
                #print(c("last user",data[i,]$user))
                #print(c("first activity", data[i,]$activity))
                #print(c("last activity", data[(i + w.size-1),]$activity))
                if (i+w.size-1 > end) {break} ## end of file reached
                ## number of observations in windows size spans the activity
                if (data[(i + w.size-1),]$activity != data[i,]$activity){break}
                ## to cover the case when exact match happened in the last iter
                ## in this case breaks here and let the loop find the new index
                ## do not check the first iteraction to avoid index 0
                if ((i > 1) & (i !=ac)) {
                        if (data[(i-1),]$activity != data[i,]$activity){break}
                }
                
                ## Get the w.size observations
                x<- as.vector(shift(data[i:(i+w.size-1),]$xaccel,0, reminder = FALSE))
                y<- as.vector(shift(data[i:(i+w.size-1),]$yaccel,0, reminder = FALSE))
                z<- as.vector(shift(data[i:(i+w.size-1),]$zaccel,0, reminder = FALSE))
                act <- bind_rows(act, as.data.frame(data[i,]$activity))
                ## Create one vector with those obsevations
                w.data <- rbind(w.data, c(as.numeric(x), as.numeric(y), as.numeric(z)))
                #print(w.data)
                #w.data <- bind_rows(w.data, as.data.frame(c(as.numeric(x), as.numeric(y), as.numeric(z))))
        }
        setTxtProgressBar(pb, i) 
        
}  
close(pb) ## close progress bar

w.data <- cbind(act,w.data)
colnames(w.data) <- c("activity",paste("xaccel", 1:w.size, sep=""), paste("yaccel", 1:w.size, sep=""), paste("zaccel", 1:w.size, sep=""))

#saveRDS(w.data, file = "w_data_window_100.RDS")
#write.csv(w.data, file = "w_data_window_100.csv")
################################################################################
## Split the windowed data into a and b datasets for 2-fold cross validation.
nrow(w.data) /2
#[1] 14777
## By looking at the data, to avoid split the same activity, we can split
## the data at row 14764 (last observation with the activity)
#a.data <- w.data[1:14764,]
#b.data <- w.data[14765:29554,]
# saveRDS(a.data, file = "a_data_window_100.RDS")
# saveRDS(b.data, file = "b_data_window_100.RDS")
a.data <- readRDS(file = "a_data_window_100.RDS")
b.data <- readRDS(file = "b_data_window_100.RDS")

 ## Check the distribution of activity
table(a.data$activity)
table(b.data$activity)
 
################################################################################
## Using Random Forest

## Define the number of trees in the random forest
nbr.tree = 51

## Train using a.data
system.time(rf.a.model <- randomForest(as.factor(activity) ~ ., data = a.data, importance = TRUE, ntree = nbr.tree))
#saveRDS(rf.a.model, file = "rf_model_a_data_window_100.RDS")
rf.a.model
#varImp(rf.a.model)
#varImpPlot(rf.a.model)

## Report the confusion matrix for b.data
rf.b.predicted <- predict(rf.a.model, b.data)
rf.b.confm <- confusionMatrix(rf.b.predicted, b.data$activity)
rf.b.confm

## Train using b.data
system.time(rf.b.model <- randomForest(as.factor(activity) ~ ., data = b.data, importance = TRUE, ntree = nbr.tree))
#saveRDS(rf.b.model, file = "rf_model_b_data_window_100.RDS")
rf.b.model
#varImp(rf.b.model)
#varImpPlot(rf.b.model)

## Report the confusion matrix for a.data
rf.a.predicted <- predict(rf.b.model, a.data)
rf.a.confm <- confusionMatrix(rf.a.predicted, a.data$activity)
rf.a.confm

##########
## Using caret for cross validation
#system.time(rf_model<-train(as.factor(activity) ~ ., data = a.data,method="rf", trControl=trainControl(method="cv",number=2), prox=TRUE,allowParallel=TRUE))
#print(rf_model)


################################################################################
## Using Neural Network
## Train using a.data
system.time(nn.a.model <- multinom (as.factor(activity) ~ . , data = a.data, MaxNWts = 2000, mxit = 500))
#saveRDS(nn.a.model, file = "nn_model_a_data_window_100.RDS")
#nn.a.model <- readRDS("nn_model_a_data_window_100.RDS")
nn.a.model

## Report the confusion matrix for b.data
nn.b.predicted <- predict(nn.a.model, newdata = b.data )
nn.b.confm <- confusionMatrix(nn.b.predicted, b.data$activity)
nn.b.confm

## Train using b.data
system.time(nn.b.model <- multinom (as.factor(activity) ~ . , data = b.data, MaxNWts = 2000, maxit = 500))
#saveRDS(nn.b.model, file = "nn_model_b_data_window_100.RDS")
#nn.b.model <- readRDS("nn_model_b_data_window_100.RDS")
nn.b.model

## Report the confusion matrix for a.data
nn.a.predicted <- predict(nn.b.model, newdata = a.data )
nn.a.confm <- confusionMatrix(nn.a.predicted, a.data$activity)
nn.a.confm
#rm(nn.model,nn.confm, nn.predicted)



################################################################################
## Using Random Forest (forest of conditional inference trees)
## Train using a.data
system.time(c.rf.a.model <- cforest(as.factor(activity) ~ ., data = a.data, controls=cforest_unbiased(ntree=nbr.tree, mtry=3)))
#saveRDS(rf.a.model, file = "rf_model_a_data_window_100.RDS")
c.rf.a.model
#varImp(rf.a.model)
#varImpPlot(rf.a.model)

## Report the confusion matrix for b.data
c.rf.b.predicted <- predict(c.rf.a.model, b.data, OOB=TRUE, type = "response")
c.rf.b.confm <- confusionMatrix(c.rf.b.predicted, b.data$activity)
c.rf.b.confm

################################################################################
## Using SVM
## Train using a.data
system.time(svm.a.model <- svm(as.factor(activity) ~ ., data = a.data ))
saveRDS(svm.a.model, file = "svm_model_a_data_window_100.RDS")
svm.a.model

## Report the confusion matrix for b.data
svm.b.predicted <- predict(svm.a.model, b.data)
svm.b.confm <- confusionMatrix(svm.b.predicted, b.data$activity)
svm.b.confm

## Train using b.data
system.time(svm.b.model <- svm(as.factor(activity) ~ ., data = b.data ))
saveRDS(svm.b.model, file = "svm_model_b_data_window_100.RDS")
svm.b.model

## Report the confusion matrix for a.data
svm.a.predicted <- predict(svm.b.model, a.data)
svm.a.confm <- confusionMatrix(svm.a.predicted, a.data$activity)
svm.a.confm

#
################################################################################
## Using fourier transform
fft.a.data <- fft(as.numeric(a.data[1,2:101]))

fft.x <- apply(a.data[,2:101], 2, fft)
fft.y <- apply(a.data[,102:201], 2, fft)
fft.z <- apply(a.data[,202:301], 2, fft)

fft.a.data <- cbind(a.data$activity, fft.x, fft.y, fft.z)
fft.a.data <- as.data.frame(fft.a.data)
#

###############################################################################
## Using Random Forest

## Define the number of trees in the random forest
nbr.tree = 51

## Train using fft.a.data
system.time(rf.a.model <- randomForest(as.factor(activity) ~ ., data = fft.a.data, importance = TRUE, ntree = nbr.tree))
#saveRDS(rf.a.model, file = "rf_model_a_data_window_100.RDS")
rf.a.model
#varImp(rf.a.model)
#varImpPlot(rf.a.model)

## Report the confusion matrix for b.data
rf.b.predicted <- predict(rf.a.model, b.data)
rf.b.confm <- confusionMatrix(rf.b.predicted, b.data$activity)
rf.b.confm

## Train using b.data
system.time(rf.b.model <- randomForest(as.factor(activity) ~ ., data = b.data, importance = TRUE, ntree = nbr.tree))
#saveRDS(rf.b.model, file = "rf_model_b_data_window_100.RDS")
rf.b.model
#varImp(rf.b.model)
#varImpPlot(rf.b.model)

## Report the confusion matrix for fft.a.data
rf.a.predicted <- predict(rf.b.model, fft.a.data)
rf.a.confm <- confusionMatrix(rf.a.predicted, fft.a.data$activity)
rf.a.confm


















##plot(fft.a.data)

#
