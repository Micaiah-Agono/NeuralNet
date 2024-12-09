
data <- read.csv('bank_note_data.csv')
head(data)
str(data)

comment <- "The data consists of 5 columns:
---variance of Wavelet Transformed image (continuous)
---skewness of Wavelet Transformed image (continuous)
---curtosis of Wavelet Transformed image (continuous)
---entropy of image (continuous)
---class (integer)

Class indicates whether or not a Bank Note was authentic
"


##
#EDA
##

library(ggplot2)
ggplot(data,aes(Entropy, Image.Curt))+geom_point()

##
#Train Test Split
##

library(caTools)
set.seed(101)

split <- sample.split(data$Class,SplitRatio = 0.7)
Train.nn <- subset(data, split==T)
Test.nn <- subset(data, split==F)
#str(Train.nn)


##
#Building the neural nets
##

library(neuralnet)
nets <- neuralnet(data=Train.nn,Class~Image.Var+Image.Skew+Image.Curt+Entropy,
                  linear.output = F, hidden=10)
plot(nets)

##
#Prediction
##

prediction.nets <- compute(nets,Test.nn[1:4])
#head(prediction.nets$net.result)

nn.pred <-sapply(prediction.nets$net.result,round)
#head(nn.pred)

#creating a confusion matrix of predicted vs real values
table(nn.pred,Test.nn$Class)

Result.nn <- "nn.pred   0   1
                    0 229   0
                    1   0 183"

#shows that the neuralnet did well


##
#Checking/Comparing with RandomForest model
##

library(randomForest)
#first convert the Class column to factor 
#because unlike neuralnets, random forest does not work with int but factor
data$Class <- factor(data$Class)
#str(data)

#re-doing sample split so i use for rf model
set.seed(101)

sampl <- sample.split(data$Class,SplitRatio = 0.7)
Train.rf <- subset(data,sampl==T)
Test.rf <- subset(data,sampl==F)

##
#Creating the randomforest
##

#rf.model <- randomForest(Class~.,data=Train.rf)
rf.model <- randomForest(Class ~ Image.Var+Image.Skew+Image.Curt+Entropy,data=Train.rf)

rf.pred <- predict(rf.model,Test.rf)
table(rf.pred,Test.rf$Class)


Result.rf <- "rf.pred   0   1
              0       227   1
              1         2 182"

#the randomForest model performed well 
#with just a little bit of the type 1 and type 2 error
#but not as much as the Neural Nets

