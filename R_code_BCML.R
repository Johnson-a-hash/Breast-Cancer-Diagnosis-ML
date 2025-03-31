#******************************************************************
# Topic : Diagnosing Breast Cancer: A Machine Learning Approach
# Author: Alberta Araba Johnson

#******************************************************************

#  CTRL + Shift + C (comment Block)

rm(list=ls())

options(scipen=10)



setwd('C:/Users/johns/OneDrive/Desktop/STATS_LEARN_A/PROJECT')



library(magrittr)
install.packages('ROCR')
library(ROCR)
library(pROC)
library(ggplot2)
library(MASS)
library(dplyr)
library(caret)
install.packages('ConfusionTableR')
library(ConfusionTableR)
install.packages('pastecs')
library(pastecs)


# Data Preparation & Descriptive Analysis
#******************************************************************

data = read.table(file = "Breast_Cancer.txt", sep=",", header = F)
# View(data)


names(data) = c("id_number", "CT", "UCSi", "UCSh", "MA", "SECS",
                "BN", "BC", "NN", "Mitoses", "Class")


data$BN[(data$BN == "?")] = NA
sum(is.na(data$BN))

str(data)

# stat.desc(data)

val = unique(data$BN[!is.na(data$BN)])
BN_mode = val[which.max(tabulate(match(data$BN, val)))]

data$BN[is.na(data$BN)] = BN_mode



# library(gridExtra)
# grid.arrange(p1, p2,p3,p4, ncol = 2)


# Data Partition

# 2 - Benign(1)
# 4 - Malignant(0) - Causes Breast Cancer

data = data[, -1]
data$BN = as.numeric(data$BN)
data$Class = (ifelse(data$Class == 4, 0, 1))


set.seed(654)
train = sample(nrow(data), 0.8*nrow(data))
train_data = data[train, ]
test_data = data[-train, ]




# Cluster Analysis (To check Important Predictors)
install.packages('ClustOfVar')
library(ClustOfVar)
library(reshape2)
library(plyr)
install.packages('Information')
library(Information)

IV = create_infotables(data = train_data, y ="Class", ncore = 2)
View(IV$Summary)



# As a general rule of thumb, IV < 0.05 means 

f = function(x) 0.05

# Information value for Each Predictor
ggplot(IV[["Summary"]], aes(y=IV, x=reorder(Variable, +IV))) +
  geom_bar(position="dodge",
           stat="identity",
           width=0.6,
           fill='blue',
           color='cyan') +
  theme_bw()+ xlab("Variables") + 
  ylab('Information Value') + coord_flip() +
  stat_function(fun = f, color = 'red')



MultiPlot(IV, IV[["Summary"]]$Variable)




# Logistic Regression
#******************************************************************

install.packages('bestglm')
library(bestglm)

# Logistic Without Subset Selection

full = glm(Class ~ ., data = train_data , family = binomial)

full.pred = (predict(full, test_data) >.5)+0

conf.full = confusionMatrix(as.factor(full.pred),
                            as.factor(test_data$Class), 
                            mode = "everything",
                            positive = '1')

full.ac = conf.full$overall[[1]]
full.sens = conf.full$byClass[[1]]
full.spec = conf.full$byClass[[2]]

binary_visualiseR(train_labels = as.factor(full.pred),
                  truth_labels= as.factor(test_data$Class),
                  class_label1 = "Malignant", 
                  class_label2 = "Benign",
                  quadrant_col1 = "lightgreen", 
                  quadrant_col2 = "#4397D2", 
                  custom_title = " ", 
                  text_col= "black",
                  positive = '1') 


pred.full = prediction(as.numeric(full.pred), as.numeric(test_data$Class))
roc.full = performance(pred.full,"tpr","fpr")

auc.full = auc(as.numeric(test_data$Class), as.numeric(full.pred))

plot(roc.full, colorize = T, lwd = 2)
abline(a = 0, b = 1, col = 'black', lwd = 2)
text(x=0.85, y=0.1, labels= paste0('AUC = ', round(auc.full*100, 5), '%'),
     pos=1, col="blue")
grid()


# Logistic With Best Subset Selection

criteria = c('AIC', 'BIC')

for (i in criteria){
  
  assign(paste('log', i, sep='.'), bestglm(train_data,
                                           IC = i,
                                           family=binomial,
                                           method = "exhaustive"))
}


var = as.factor(1:9)

md_df1 = data.frame(var, log.AIC$Subsets$AIC[-1], rep('AIC', 9))
colnames(md_df1) = c('var', 'value', 'criteria')

md_df2 = data.frame(var, log.BIC$Subsets$BIC[-1], rep('BIC', 9))
colnames(md_df2) = c('var', 'value', 'criteria')

df_plot = rbind(md_df1, md_df2)

ggplot(df_plot, aes(x=var, y=value, fill=criteria, group=criteria)) +
  geom_line(size=1) + theme_bw() + geom_point(size = 4, shape = 21) +
  xlab("Number of Variables") + ylab('Value')+
  geom_segment(aes(x = which.min(df_plot[(df_plot$criteria) %in% 'BIC', ]$value),
                   y = 108, xend = 4, yend = 116), size = 1.1)+
  geom_segment(aes(x = which.min(df_plot[(df_plot$criteria) %in% 'AIC', ]$value),
                   y = 86, xend = 6, yend = 94), size = 1.1)+
  theme(axis.title.x = element_text(face="bold"),
        axis.title.y = element_text(face="bold"))



# Prediction and Confusion Matrix for Subset Models

aic.pred = (predict(log.AIC$BestModel, test_data) >.5)+0
conf.aic = confusionMatrix(as.factor(aic.pred),
                           as.factor(test_data$Class), 
                           mode = "everything",
                           positive = '1')

aic.ac = conf.aic$overall[[1]]
aic.sens = conf.aic$byClass[[1]]
aic.spec = conf.aic$byClass[[2]]


t.aic = conf.aic$table
dimnames(t.aic)[[1]] = c("Benign", "Malignant")
dimnames(t.aic)[[2]] = c("Benign", "Malignant")



bic.pred = (predict(log.BIC$BestModel, test_data) >.5)+0
conf.bic = confusionMatrix(as.factor(bic.pred),
                           as.factor(test_data$Class), 
                           mode = "everything",
                           positive = '1')

bic.ac = conf.bic$overall[[1]]
bic.sens = conf.bic$byClass[[1]]
bic.spec = conf.bic$byClass[[2]]


t.bic = conf.bic$table
dimnames(t.bic)[[1]] = c("Benign", "Malignant")
dimnames(t.bic)[[2]] = c("Benign", "Malignant")


par(mfrow=c(1,2))

fourfoldplot(t.aic,
             color = c("#B22222","lightgreen"),
             conf.level = 0, margin = 1,
             main = 'Model Based on AIC') + 
  text(-0.4,0.4, "TN", cex=1)  + 
  text(0.4, -0.5, "TP", cex=1) + 
  text(0.5,0.4, "FN", cex=1)   + 
  text(-0.4, -0.5, "FP", cex=1)

fourfoldplot(t.bic,
             color = c("#B22222", "lightgreen"),
             conf.level = 0, margin = 1,
             main = 'Model Based on BIC') + 
  text(-0.4,0.4, "TN", cex=1)  + 
  text(0.4, -0.5, "TP", cex=1) + 
  text(0.5,0.4, "FN", cex=1)   + 
  text(-0.4, -0.5, "FP", cex=1)



# KNN Classification
#***********************************************************************





# Naive-Baye's Classification
#******************************************************************

library(e1071)

nb.fit = naiveBayes(Class ~ ., data = train_data)

nb.pred = predict(nb.fit, test_data)


conf.nb = confusionMatrix(as.factor(nb.pred),
                           as.factor(test_data$Class),
                           mode = "everything",
                           positive = "1")

nb.ac = conf.nb$overall[[1]]
nb.sens = conf.nb$byClass[[1]]
nb.spec = conf.nb$byClass[[2]]


binary_visualiseR(train_labels = as.factor(nb.pred),
                  truth_labels= as.factor(test_data$Class),
                  class_label1 = "Benign", 
                  class_label2 = "Malignant",
                  quadrant_col1 = "lightgreen", 
                  quadrant_col2 = "#4397D2", 
                  custom_title = " ", 
                  text_col= "black",
                  positive = '1') 




# Discriminant Analysis
#******************************************************************

library(MASS)

# Linear Discriminant Analysis
lda.fit = lda(Class~ ., data = train_data)
lda.pred = predict(lda.fit, test_data)$class


conf.lda = confusionMatrix(as.factor(lda.pred),
                           as.factor(test_data$Class),
                           mode = "everything",
                           positive = "1")

lda.ac = conf.lda$overall[[1]]
lda.sens = conf.lda$byClass[[1]]
lda.spec = conf.lda$byClass[[2]]


binary_visualiseR(train_labels = as.factor(lda.pred),
                  truth_labels= as.factor(test_data$Class),
                  class_label1 = "Benign", 
                  class_label2 = "Malignant",
                  quadrant_col1 = "lightgreen", 
                  quadrant_col2 = "#4397D2", 
                  custom_title = " ", 
                  text_col= "black",
                  positive = '1') 



# Quadratic Discriminant Analysis
qda.fit = qda(Class~ ., data = train_data)
qda.pred = predict(qda.fit, test_data)$class


conf.qda = confusionMatrix(as.factor(qda.pred),
                           as.factor(test_data$Class),
                           mode = "everything",
                           positive = "1")

qda.ac = conf.qda$overall[[1]]
qda.sens = conf.qda$byClass[[1]]
qda.spec = conf.qda$byClass[[2]]


binary_visualiseR(train_labels = as.factor(qda.pred),
                  truth_labels= as.factor(test_data$Class),
                  class_label1 = "Benign", 
                  class_label2 = "Malignant",
                  quadrant_col1 = "lightgreen", 
                  quadrant_col2 = "#4397D2", 
                  custom_title = " ", 
                  text_col= "black",
                  positive = '1') 




# Random Forest Classifier
#******************************************************************




# Decision Tree Calssifier
#******************************************************************

install.packages("tree")
library(tree)

tree.fit = tree(Class ~ ., data = train_data)
summary(tree.fit)$used

plot(tree.fit)
text(tree.fit,cex=1,col="blue",pretty=0)

library(rpart)

install.packages('rpart.plot')
library(rpart.plot)


tree1 = rpart(Class ~ ., train_data,
              control=rpart.control(cp=.0001))

best_tree = tree1$cptable[which.min(tree1$cptable[,"xerror"]),"CP"]

prp(tree1)


prp(tree1,
    faclen=0, #use full names for factor labels
    extra=1, #display number of observations for each terminal node
    roundint=F, #don't round to integers in output
    digits=5) #display 5 decimal places in output


rpart.plot(tree1,type=3,under = TRUE,
           space = 0,tweak=1.8,under.cex = 1,
           fallen.leaves = TRUE,branch.lty = 3)

# CV for Decision Tree
set.seed(123)

d = rpart(Class ~ ., train_data, cp = .01,parms=list(split="gini"),
      minsplit=20, minbucket=7, maxdepth=10)



accuracy_tune <- function(fit) {
  predict_unseen <- predict(fit, test_data, type = 'class')
  table_mat <- table(test_data$Class, predict_unseen)
  accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
  accuracy_Test
}


control = rpart.control(minsplit = 4,
                         minbucket = round(5/3),
                         maxdepth = 3,
                         cp = 0)

tune_fit = rpart(Class ~., data = train_data,
                  method = 'class',
                  control = control)

accuracy_tune(tune_fit)

summary(tune_fit)








# Artificial Neural Network
#******************************************************************


set.seed(000)
install.packages("neuralnet")
library("neuralnet")
library("MASS")





# Gradient Boosting Classifier
#******************************************************************
install.packages('gbm')
library(gbm)

set.seed(4325)
model.gbm = gbm(Class ~ .,
                data = train_data,
                distribution = "multinomial",
                cv.folds = 10,
                shrinkage = .01,
                n.minobsinnode = 10,
                n.trees = 500,
                n.cores = NULL,
                interaction.depth = 3)

summary(model.gbm)

# influence plot
ggplot(summary(model.gbm),aes(y=rel.inf,x=reorder(var, + rel.inf))) +
  geom_bar(position="dodge", stat="identity",
           width=0.6, fill='blue', color='red') +
  theme_bw()+ xlab("Variables") + 
  ylab('Relative Influence') + coord_flip() 

best.iter = gbm.perf(model.gbm, method="cv")
grid()

gbm.pred = predict.gbm(object = model.gbm,
                        newdata = test_data,
                        n.trees = 500,
                        cv.folds=10,
                        type = "response")

gbm.pred = colnames(gbm.pred)[apply(gbm.pred, 1, which.max)]

conf.gbm = confusionMatrix(as.factor(gbm.pred),
                          as.factor(test_data$Class),
                          mode = "everything",
                          positive = "1")

gbm.ac = conf.gbm$overall[[1]]
gbm.sens = conf.gbm$byClass[[1]]
gbm.spec = conf.gbm$byClass[[2]]

binary_visualiseR(train_labels = as.factor(gbm.pred),
                  truth_labels= as.factor(test_data$Class),
                  class_label1 = "Benign", 
                  class_label2 = "Malignant",
                  quadrant_col1 = "lightgreen", 
                  quadrant_col2 = "#4397D2", 
                  custom_title = " ", 
                  text_col= "black",
                  positive = '1')


pred.xg = prediction(as.numeric(gbm.pred), as.numeric(test_data$Class))
roc.xg = performance(pred.full,"tpr","fpr")

auc.xg = auc(as.numeric(test_data$Class), as.numeric(gbm.pred))

plot(roc.xg, colorize = T, lwd = 2)
abline(a = 0, b = 1, col = 'black', lwd = 2)
text(x=0.85, y=0.1, labels= paste0('AUC = ', round(auc.xg*100, 5), '%'),
     pos=1, col="blue")
grid()
