

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

## Assign the column names
colnames(data) <- c("user","activity","timestamp","xaccel","yaccel","zaccel")
#colnames(test.data) <- c("user","activity","timestamp","xaccel","yaccel","zaccel")

## Drop the last column that is wrongly interpreted as column by R
## because each line ends in ","
data <- data[, -7]

######################################################################
## Check which are the indices when user changes
#user.indexes <- which(data$user != lag(data$user))
## input data

act.indexes <- which(data$activity != lag(data$activity))
## Add the firt onbservation index
act.indexes <- c(1,  act.indexes)

## Create the windowed data frame.
w.data <-data.frame()
act <- data.frame()

w.size = 100 ## window size
## widowed dataset
w.file.name <- paste("w_data_window_", w.size, ".RDS", sep="")
## A data set
a.file.name <- paste("a_data_window_", w.size, ".RDS", sep="")
## B data set
b.file.name <- paste("b_data_window_", w.size, ".RDS", sep="")

end = nrow(data) ## end of file

## Create the new vectors using w.size observations
pb <- txtProgressBar(min = 0, max = end, style = 3) ## progress bar

## Loop for activity. I takes about 1 hour
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

saveRDS(w.data, file = w.file.name)
#write.csv(w.data, file = "w_data_window_100.csv")

################################################################################
## Split the windowed data into a and b datasets for 2-fold cross validation.
nrow(w.data) / 2
#[1] 7322.5

## Let's check where activities change near to the middle of the dataset
act.indexex.new <- which(w.data$activity != lag(w.data$activity))
## By looking at the result the closest change is at 7323 so 7322 is chosen as the
## last observation of the activity for a dataset.

a.data <- w.data[1:7322,]
b.data <- w.data[7323:nrow(w.data),]
#saveRDS(a.data, file = a.file.name)
#saveRDS(b.data, file = b.file.name)

## Check the distribution of activity
table(a.data$activity)
table(b.data$activity)

################################################################################
## Using fourier transform
## Load the data sets with window = 100
a.data <- readRDS(file = "a_data_window_100.RDS")
b.data <- readRDS(file = "b_data_window_100.RDS")

## The step takes about 30 minutes for each data set.
## Apply the Fourier transform and get the absolute values for a data set.
pb <- txtProgressBar(min = 0, max = nrow(a.data), style = 3) ## progress bar
for (i in 1:nrow(a.data)){
        a.data[i, 2:101] <- fft(as.numeric(a.data[i, 2:101]))
        a.data[i, 102:201] <- fft(as.numeric(a.data[i, 102:201]))
        a.data[i, 202:301] <- fft(as.numeric(a.data[i, 202:301]))
        
        a.data[i, 2:101] <- abs(as.numeric(a.data[i, 2:101]))
        a.data[i, 102:201] <- abs(as.numeric(a.data[i, 102:201]))
        a.data[i, 202:301] <- abs(as.numeric(a.data[i, 202:301]))
        
        setTxtProgressBar(pb, i)
}
close(pb) ## close progress bar
saveRDS(a.data, file = "a_data_window_100_fft_abs.RDS")

## Apply the Fourier transform and get the absolute values for b data set.
pb <- txtProgressBar(min = 0, max = nrow(b.data), style = 3) ## progress bar
for (i in 1:nrow(b.data)){
        b.data[i, 2:101] <- fft(as.numeric(b.data[i, 2:101]))
        b.data[i, 102:201] <- fft(as.numeric(b.data[i, 102:201]))
        b.data[i, 202:301] <- fft(as.numeric(b.data[i, 202:301]))
        
        b.data[i, 2:101] <- abs(as.numeric(b.data[i, 2:101]))
        b.data[i, 102:201] <- abs(as.numeric(b.data[i, 102:201]))
        b.data[i, 202:301] <- abs(as.numeric(b.data[i, 202:301]))
        
        setTxtProgressBar(pb, i)
}
close(pb) ## close progress bar
saveRDS(b.data, file = "b_data_window_100_fft_abs.RDS")

## Plot some data on frequency domain
time <- 5 # measuring time interval (s). 100 (window) * 50 miliseconds (sample)
# extract magnitude
magn <- Mod(a.data[1, 2:101]) ## for x
## To calculated the frequencies, simply take (or generate) the index vector (1 
## to length(magnitude vector) and divide by the length of the data block (in sec).
# generate x-axis with frequencies
x.axis <- 1:length(magn)/time

# plot magnitudes against frequencies
plot(x=x.axis,y=magn.1,type="l")

################################################################################
## Using Random Forest
set.seed(1)

# ## Load the windowed data Fourier transformed data
# a.data <- readRDS(file = "a_data_window_100_fft_abs.RDS")
# b.data <- readRDS(file = "b_data_window_100_fft_abs.RDS")
# 
# ## Load the saved random forest models
# rf.a.model <- readRDS("rf_model_a_data_window_100_fft.RDS")
# rf.b.model <- readRDS("rf_model_b_data_window_100_fft.RDS")

## Define the number of trees in the random forest
nbr.tree = 51

## Train using a.data
system.time(rf.a.model <- randomForest(as.factor(activity) ~ ., data = a.data, importance = TRUE, ntree = nbr.tree))
# saveRDS(rf.a.model, file = "rf_model_a_data_window_100_fft.RDS")
rf.a.model
#varImp(rf.a.model)
#varImpPlot(rf.a.model)

## Report the confusion matrix for b.data
rf.b.predicted <- predict(rf.a.model, b.data)
rf.b.confm <- confusionMatrix(rf.b.predicted, b.data$activity)
rf.b.confm

## Train using b.data
system.time(rf.b.model <- randomForest(as.factor(activity) ~ ., data = b.data, importance = TRUE, ntree = nbr.tree))
# saveRDS(rf.b.model, file = "rf_model_b_data_window_100_fft.RDS")
rf.b.model
#varImp(rf.b.model)
#varImpPlot(rf.b.model)

## Report the confusion matrix for a.data
rf.a.predicted <- predict(rf.b.model, a.data)
rf.a.confm <- confusionMatrix(rf.a.predicted, a.data$activity)
rf.a.confm
################################################################################
## Feature Engineering
# Based on this finding, this work extracts two kinds of time domain features from 
# the accelerometer sensor data:  SSFs: mean, standard deviation, correlation, and 
# signal magnitude area;  coefficients of time series analysis, including 
# autoregressive (AR) analysis, and moving average (MA) analysis.

## Load the data sets with window = 100
a.data <- readRDS(file = "a_data_window_100.RDS")
b.data <- readRDS(file = "b_data_window_100.RDS")

## Calculate the mean, sd, min, values for a data
x.mean <- apply(a.data[,2:101], 1, mean)
y.mean <- apply(a.data[,102:201], 1, mean)
z.mean <- apply(a.data[,202:301], 1, mean)

x.sd <- apply(a.data[,2:101], 1, sd)
y.sd <- apply(a.data[,102:201], 1, sd)
z.sd <- apply(a.data[,202:301], 1, sd)

x.min <- apply(a.data[,2:101], 1, min)
y.min <- apply(a.data[,102:201], 1, min)
z.min <- apply(a.data[,202:301], 1, min)

x.max <- apply(a.data[,2:101], 1, max)
y.max <- apply(a.data[,102:201], 1, max)
z.max <- apply(a.data[,202:301], 1, max)

a.data <- cbind(a.data, x.mean, x.sd, x.min, x.max, y.mean, y.sd, y.min, y.max, z.mean, z.sd, z.min, z.max)

## Calculate the mean, sd, min, values for b data
x.mean <- apply(b.data[,2:101], 1, mean)
y.mean <- apply(b.data[,102:201], 1, mean)
z.mean <- apply(b.data[,202:301], 1, mean)

x.sd <- apply(b.data[,2:101], 1, sd)
y.sd <- apply(b.data[,102:201], 1, sd)
z.sd <- apply(b.data[,202:301], 1, sd)

x.min <- apply(b.data[,2:101], 1, min)
y.min <- apply(b.data[,102:201], 1, min)
z.min <- apply(b.data[,202:301], 1, min)

x.max <- apply(b.data[,2:101], 1, max)
y.max <- apply(b.data[,102:201], 1, max)
z.max <- apply(b.data[,202:301], 1, max)

b.data <- cbind(b.data, x.mean, x.sd, x.min, x.max, y.mean, y.sd, y.min, y.max, z.mean, z.sd, z.min, z.max)

################################################################################
## PCA

## Load the windowed data Fourier transformed data
a.data <- readRDS(file = "a_data_window_100_fft_abs.RDS")
b.data <- readRDS(file = "b_data_window_100_fft_abs.RDS")

# ## Load the data sets with window = 100
# a.data <- readRDS(file = "a_data_window_100.RDS")
# b.data <- readRDS(file = "b_data_window_100.RDS")

## Calculate the PCA.
pca.a <- prcomp(a.data[, 2:301], scale = TRUE)
pca.b <- prcomp(b.data[, 2:301], scale = TRUE)
# The plot method returns a plot of the variances (y-axis) associated
# with the PCs (x-axis). The Figure below is useful to decide how many
# PCs to retain for further analysis. 
#plot(pca.a, type = "l", main = "PC x Variances", col = 2)

## A better approach is to check the cumulative proportion.
summary(pca.a)
summary(pca.b)
## In this case 104 PCs represent 100% of the variance for a data set
## In case of b data set we need 153 PC. 104 PCS gives a better result during 
## cross validation so 104 is chosen.

## Using only the n principal components. 
n <- 104

## This is just to print the parameters to be used in  the formula of random forest.
PCs <-"PC1"
for (i in 2:n) {
        PCs <- paste(PCs, "+PC", i, sep="")

}
print(PCs)

## Prepare the data sets with only the PCs from PCA
pca.a.comp <- predict(pca.a, newdata = a.data)[,1:n]
pca.a.data <- as.data.frame(cbind(a.data, pca.a.comp))
## Get the b data set with the PC
pca.b.comp <- predict(pca.a, newdata = b.data)[,1:n]
pca.b.data <- data.frame(b.data, pca.b.comp)

## Run Random Forest on PCA.
## Train on a data set
set.seed(1)
nbr.tree = 51
system.time(rf.a.model <- randomForest(as.factor(activity) ~ PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10+PC11+PC12+PC13+PC14+PC15+PC16+PC17+PC18+PC19+PC20+PC21+PC22+PC23+PC24+PC25+PC26+PC27+PC28+PC29+PC30+PC31+PC32+PC33+PC34+PC35+PC36+PC37+PC38+PC39+PC40+PC41+PC42+PC43+PC44+PC45+PC46+PC47+PC48+PC49+PC50+PC51+PC52+PC53+PC54+PC55+PC56+PC57+PC58+PC59+PC60+PC61+PC62+PC63+PC64+PC65+PC66+PC67+PC68+PC69+PC70+PC71+PC72+PC73+PC74+PC75+PC76+PC77+PC78+PC79+PC80+PC81+PC82+PC83+PC84+PC85+PC86+PC87+PC88+PC89+PC90+PC91+PC92+PC93+PC94+PC95+PC96+PC97+PC98+PC99+PC100+PC101+PC102+PC103+PC104, data = pca.a.data, importance = TRUE, ntree = nbr.tree))
rf.a.model
## Report the confusion matrix for b.data
rf.b.predicted <- predict(rf.a.model, pca.b.data)
rf.b.confm <- confusionMatrix(rf.b.predicted, pca.b.data$activity)
rf.b.confm

## Prepare the data sets with only the PCs from PCA
pca.b.comp <- predict(pca.b, newdata = b.data)[,1:n]
pca.b.data <- as.data.frame(cbind(b.data, pca.b.comp))
## Get the a data set with the PC
pca.a.comp <- predict(pca.b, newdata = a.data)[,1:n]
pca.a.data <- data.frame(a.data, pca.a.comp)

## Run Random Forest on PCA.
##Train on b data set
system.time(rf.b.model <- randomForest(as.factor(activity) ~ PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10+PC11+PC12+PC13+PC14+PC15+PC16+PC17+PC18+PC19+PC20+PC21+PC22+PC23+PC24+PC25+PC26+PC27+PC28+PC29+PC30+PC31+PC32+PC33+PC34+PC35+PC36+PC37+PC38+PC39+PC40+PC41+PC42+PC43+PC44+PC45+PC46+PC47+PC48+PC49+PC50+PC51+PC52+PC53+PC54+PC55+PC56+PC57+PC58+PC59+PC60+PC61+PC62+PC63+PC64+PC65+PC66+PC67+PC68+PC69+PC70+PC71+PC72+PC73+PC74+PC75+PC76+PC77+PC78+PC79+PC80+PC81+PC82+PC83+PC84+PC85+PC86+PC87+PC88+PC89+PC90+PC91+PC92+PC93+PC94+PC95+PC96+PC97+PC98+PC99+PC100+PC101+PC102+PC103+PC104, data = pca.b.data, importance = TRUE, ntree = nbr.tree))
rf.b.model
## Report the confusion matrix for a.data
rf.a.predicted <- predict(rf.b.model, pca.a.data)
rf.a.confm <- confusionMatrix(rf.a.predicted, pca.a.data$activity)
rf.a.confm













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
## For time domain
#system.time(svm.a.model <- svm(as.factor(activity) ~ ., data = a.data ))
#saveRDS(svm.a.model, file = "svm_model_a_data_window_100.RDS")
svm.a.model <- readRDS("svm_model_a_data_window_100.RDS")

svm.a.model

## Report the confusion matrix for b.data
svm.b.predicted <- predict(svm.a.model, b.data)
svm.b.confm <- confusionMatrix(svm.b.predicted, b.data$activity)
svm.b.confm

## Train using b.data
#system.time(svm.b.model <- svm(as.factor(activity) ~ ., data = b.data ))
#saveRDS(svm.b.model, file = "svm_model_b_data_window_100.RDS")
svm.b.model

## Report the confusion matrix for a.data
svm.a.predicted <- predict(svm.b.model, a.data)
svm.a.confm <- confusionMatrix(svm.a.predicted, a.data$activity)
svm.a.confm

#
