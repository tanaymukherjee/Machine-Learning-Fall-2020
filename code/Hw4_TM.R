rm(list = ls())    #delete objects
cat("\014")        #clear console

library(keras)
library(tensorflow)
library(tidyverse)
library(glmnet) 
library(latex2exp)

p                   =     2500
imdb                =     dataset_imdb(num_words = p, skip_top = 00) #, skip_top = 10
train_data          =     imdb$train$x
train_labels        =     imdb$train$y
test_data           =     imdb$test$x
test_labels         =     imdb$test$y

numberWords.train   =   max(sapply(train_data, max))
numberWords.test    =   max(sapply(test_data, max))

vectorize_sequences <- function(sequences, dimension = p) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

X.train          =        vectorize_sequences(train_data)
X.test           =        vectorize_sequences(test_data)

y.train          =        as.numeric(train_labels)
n.train          =        length(y.train)
y.test           =        as.numeric(test_labels)
n.test           =        length(y.test)

# Sampling
train_sample_ind <- c(which(y.train==0)[1:12500],which(y.train==1)[1:4000])
test_sample_ind <- c(which(y.test==0)[1:12500],which(y.test==1)[1:4000])

X.train <- X.train[train_sample_ind, ]
X.test  <- X.test[test_sample_ind , ]
y.train <- y.train[train_sample_ind]
y.test  <- y.test[test_sample_ind]

# ----


## Lasso regression

Lasso = cv.glmnet(x = X.train, y=y.train, family = "binomial", alpha = 1, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc")

## in the case of AUC measure
## lasso$lambda[which.max(lasso$cvm)] is the same as lasso$lambda.min

lasso_best = glmnet(x = X.train, y=y.train, lambda = Lasso$lambda.min, family = "binomial", alpha = 1, intercept = TRUE, standardize = FALSE)

beta0.hat = lasso_best$a0
beta.hat = as.vector(lasso_best$beta)

thrs                    =        0.5
prob.train              =        exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat  ))
y.hat.train             =        ifelse(prob.train > thrs, 1, 0) #table(y.hat.train, y.train)
FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
P.train                 =        sum(y.train==1) # total positives in the data
N.train                 =        sum(y.train==0) # total negatives in the data
FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
typeI.err.train         =        FPR.train
typeII.err.train        =        1 - TPR.train

print(paste( "train: err        = ", sprintf("%.2f" , mean(y.train != y.hat.train))))
print(paste( "train: typeI.err  = ", sprintf("%.2f" , typeI.err.train)))
print(paste( "train: typeII.err = ", sprintf("%.2f" , typeII.err.train)))


pi.hat                  =        sum(y.train)/n.train
y.hat.train.rand        =        rbinom(n.train, 1, pi.hat) # draws n.train independent binary r.v.s that are 1 with prob. pi.hat 
FP.train                =        sum(y.train[y.hat.train.rand ==1] == 0) # false positives = negatives in the data that were predicted as positive
TP.train                =        sum(y.hat.train.rand[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
P.train                 =        sum(y.train==1) # total positives in the data
N.train                 =        sum(y.train==0) # total negatives in the data

FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
typeI.err.train         =        FPR.train
typeII.err.train        =        1 - TPR.train
print(paste("---------------random predictor --------------------"))
print(paste( "train: err        = ", sprintf("%.2f" , mean(y.train != y.hat.train.rand))))
print(paste( "train: typeI.err  = ", sprintf("%.2f" , typeI.err.train)))
print(paste( "train: typeII.err = ", sprintf("%.2f" , typeII.err.train)))




prob.test               =        exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
y.hat.test              =        ifelse(prob.test > thrs,1,0) #table(y.hat.test, y.test)  
FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
P.test                  =        sum(y.test==1) # total positives in the data
N.test                  =        sum(y.test==0) # total negatives in the data
TN.test                 =        sum(y.hat.test[y.test==0] == 0)# negatives in the data that were predicted as negatives
FPR.test                =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
TPR.test                =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity = recall
typeI.err.test          =        FPR.test
typeII.err.test         =        1 - TPR.test
print(paste("--------------- --------------------"))
print(paste( "test: err        = ", sprintf("%.2f" , mean(y.test != y.hat.test))))
print(paste( "test: typeI.err  = ", sprintf("%.2f" , typeI.err.test)))
print(paste( "test: typeII.err = ", sprintf("%.2f" , typeII.err.test)))




str(train_data[[1]])

word_index                   =     dataset_imdb_word_index() #word_index is a named list mapping words to an integer index
reverse_word_index           =     names(word_index) # Reverses it, mapping integer indices to words
names(reverse_word_index)    =     word_index

review_index                 =     1
decoded_review <- sapply(train_data[[review_index]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
}) 



cat(decoded_review)

library(dplyr)
obh                    =       order(beta.hat) 
mw                     =       50
word.index.negatives   =       obh[1:mw]
word.index.positives   =       obh[(p-(mw-1)):p]


negative.Words         =       sapply(word.index.negatives, function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
cat(negative.Words)



positive.Words         =       sapply(word.index.positives, function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
cat(positive.Words)


# -- top  5 Pos and Neg words
cat(rev(tail(positive.Words,5)),sep = ", ")
cat(head(negative.Words,5),sep = ", ")


## ROC
thrs_seq   <- c(seq(0,1, by = 0.01))
FPR_train <- TPR_train <- FPR_test <- TPR_test <- rep(0, length(thrs_seq))

prob.train                =        exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat))
prob.test                 =        exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat))

for (i in 1:length(thrs_seq)){
  thrs                    = thrs_seq[i]
  print(paste('For the threshold sequence:',sprintf("%.2f" , thrs_seq[i])))
  
  # for training set
  y.hat.train             =        ifelse(prob.train > thrs, 1, 0) #table(y.hat.train, y.train)
  FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
  P.train                 =        sum(y.train==1) # total positives in the data
  N.train                 =        sum(y.train==0) # total negatives in the data
  FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
  TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity = power
  typeI.err.train         =        FPR.train
  typeII.err.train        =        1 - TPR.train
  FPR_train[i]            =        typeI.err.train
  TPR_train[i]            =        1 - typeII.err.train
  print(paste('FPR for training set is',FPR_train[i]))
  print(paste('TPR for training set is',TPR_train[i]))
  
  # for test set
  y.hat.test              =        ifelse(prob.test > thrs,1,0) #table(y.hat.test, y.test)  
  FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
  P.test                  =        sum(y.test==1) # total positives in the data
  N.test                  =        sum(y.test==0) # total negatives in the data
  TN.test                 =        sum(y.hat.test[y.test==0] == 0)# negatives in the data that were predicted as negatives
  FPR.test                =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
  TPR.test                =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity = recall
  typeI.err.test          =        FPR.test
  typeII.err.test         =        1 - TPR.test
  FPR_test[i]             =        typeI.err.test
  TPR_test[i]             =        1 - typeII.err.test
  print(paste('FPR for test set is',FPR_test[i]))
  print(paste('TPR for test set is',TPR_test[i]))
}

train = data.frame(FPR = FPR_train, TPR = TPR_train, Set = 'Train', Threshold = thrs_seq)
test  = data.frame(FPR = FPR_test,  TPR = TPR_test,  Set = 'Test',  Threshold = thrs_seq)
df <- rbind(train, test)

# Using colAUC method in catools library to get the AUC value
library(caTools)

# ROC for training set
AUC_train = colAUC(prob.train, y.train, plotROC = F)[1]

# ROC for test set
AUC_test = colAUC(prob.test, y.test, plotROC = F)[1]

# Plot the ROC curves
ggplot(data = df, aes(x=FPR, y = TPR, col = Set))+
  geom_line(show.legend = T)+
  labs(title = 'ROC Curves for Training and Test from Lasso Logistic Regression', x = 'False Positive Rate (FPR)', y = 'True Positive Rate (TPR)' ) +
  annotate(geom="text", 
           x=c(0.5,0.5), 
           y=c(0.4,0.5), 
           label=c(paste('AUC of Test: ',round(AUC_test, 3)), paste('AUC of Train: ',round(AUC_train, 3))), 
           color=c('red', 'blue'))+
  scale_color_manual(values=c('red', 'blue'))


## For thr = 0.5
print( paste('Type 1 Error for Training Set:',df[df$Threshold == 0.5 & df$Set=='Train', 'FPR'] ))
print( paste('Type 2 Error for Training Set:',(1 - df[df$Threshold == 0.5 & df$Set=='Train', 'TPR'] )))
print( paste('Type 1 Error for Test Set:',df[df$Threshold == 0.5 & df$Set=='Test', 'FPR'] ))
print( paste('Type 2 Error for Test Set:',(1 - df[df$Threshold == 0.5 & df$Set=='Test', 'TPR'] )))




## Optimal error loss
df_train = df[df$Set == 'Train', ]
df_test = df[df$Set == 'Test', ]

df_train[which.min(abs(df_train$FPR - (1-df_train$TPR))), 'Threshold']
df_test [which.min(abs(df_test$FPR - (1-df_test$TPR))), 'Threshold']

## for train
df_train[which.min(abs(df_train$FPR - (1-df_train$TPR))), ]
## for test
df_test [which.min(abs(df_test$FPR - (1-df_test$TPR))), ]





####################################


## Ridge regression

Ridge = cv.glmnet(x = X.train, y=y.train, family = "binomial", alpha = 0, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc")

## in the case of AUC measure
## lasso$lambda[which.max(lasso$cvm)] is the same as lasso$lambda.min

ridge_best = glmnet(x = X.train, y=y.train, lambda = Ridge$lambda[which.max(Ridge$cvm)], family = "binomial", alpha = 0, intercept = TRUE, standardize = FALSE)

beta0.hat = ridge_best$a0
beta.hat = as.vector(ridge_best$beta)

thrs                    =        0.5
prob.train              =        exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat  ))
y.hat.train             =        ifelse(prob.train > thrs, 1, 0) #table(y.hat.train, y.train)
FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
P.train                 =        sum(y.train==1) # total positives in the data
N.train                 =        sum(y.train==0) # total negatives in the data
FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
typeI.err.train         =        FPR.train
typeII.err.train        =        1 - TPR.train

print(paste( "train: err        = ", sprintf("%.2f" , mean(y.train != y.hat.train))))
print(paste( "train: typeI.err  = ", sprintf("%.2f" , typeI.err.train)))
print(paste( "train: typeII.err = ", sprintf("%.2f" , typeII.err.train)))


pi.hat                  =        sum(y.train)/n.train
y.hat.train.rand        =        rbinom(n.train, 1, pi.hat) # draws n.train independent binary r.v.s that are 1 with prob. pi.hat 
FP.train                =        sum(y.train[y.hat.train.rand ==1] == 0) # false positives = negatives in the data that were predicted as positive
TP.train                =        sum(y.hat.train.rand[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
P.train                 =        sum(y.train==1) # total positives in the data
N.train                 =        sum(y.train==0) # total negatives in the data

FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
typeI.err.train         =        FPR.train
typeII.err.train        =        1 - TPR.train
print(paste("---------------random predictor --------------------"))
print(paste( "train: err        = ", sprintf("%.2f" , mean(y.train != y.hat.train.rand))))
print(paste( "train: typeI.err  = ", sprintf("%.2f" , typeI.err.train)))
print(paste( "train: typeII.err = ", sprintf("%.2f" , typeII.err.train)))




prob.test               =        exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
y.hat.test              =        ifelse(prob.test > thrs,1,0) #table(y.hat.test, y.test)  
FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
P.test                  =        sum(y.test==1) # total positives in the data
N.test                  =        sum(y.test==0) # total negatives in the data
TN.test                 =        sum(y.hat.test[y.test==0] == 0)# negatives in the data that were predicted as negatives
FPR.test                =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
TPR.test                =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity = recall
typeI.err.test          =        FPR.test
typeII.err.test         =        1 - TPR.test
print(paste("--------------- --------------------"))
print(paste( "test: err        = ", sprintf("%.2f" , mean(y.test != y.hat.test))))
print(paste( "test: typeI.err  = ", sprintf("%.2f" , typeI.err.test)))
print(paste( "test: typeII.err = ", sprintf("%.2f" , typeII.err.test)))


str(train_data[[1]])

word_index                   =     dataset_imdb_word_index() #word_index is a named list mapping words to an integer index
reverse_word_index           =     names(word_index) # Reverses it, mapping integer indices to words
names(reverse_word_index)    =     word_index

review_index                 =     1
decoded_review <- sapply(train_data[[review_index]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
}) 
# Decodes the review. 
# Note that the indices are offset by 3 because 0, 1, and 2 are reserved indices 
# for “padding,” “start of sequence,” and “unknown.”


cat(decoded_review)

library(dplyr)
obh                    =       order(beta.hat) 
mw                     =       50
word.index.negatives   =       obh[1:mw]
word.index.positives   =       obh[(p-(mw-1)):p]


negative.Words         =       sapply(word.index.negatives, function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
cat(negative.Words)



positive.Words         =       sapply(word.index.positives, function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
cat(positive.Words)


# -- Top 5 Pos and Neg words
cat(rev(tail(positive.Words,5)),sep = ", ")
cat(head(negative.Words,5),sep = ", ")


## ROC
thrs_seq   <- c(seq(0,1, by = 0.01))
FPR_train <- TPR_train <- FPR_test <- TPR_test <- rep(0, length(thrs_seq))

prob.train                =        exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat))
prob.test                 =        exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat))

for (i in 1:length(thrs_seq)){
  thrs                    = thrs_seq[i]
  print(paste('For the threshold sequence:',sprintf("%.2f" , thrs_seq[i])))
  
  # for training set
  y.hat.train             =        ifelse(prob.train > thrs, 1, 0) #table(y.hat.train, y.train)
  FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
  P.train                 =        sum(y.train==1) # total positives in the data
  N.train                 =        sum(y.train==0) # total negatives in the data
  FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
  TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity = power
  typeI.err.train         =        FPR.train
  typeII.err.train        =        1 - TPR.train
  FPR_train[i]            =        typeI.err.train
  TPR_train[i]            =        1 - typeII.err.train
  print(paste('FPR for training set is',FPR_train[i]))
  print(paste('TPR for training set is',TPR_train[i]))
  
  # for test set
  y.hat.test              =        ifelse(prob.test > thrs,1,0) #table(y.hat.test, y.test)  
  FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
  P.test                  =        sum(y.test==1) # total positives in the data
  N.test                  =        sum(y.test==0) # total negatives in the data
  TN.test                 =        sum(y.hat.test[y.test==0] == 0)# negatives in the data that were predicted as negatives
  FPR.test                =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
  TPR.test                =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity = recall
  typeI.err.test          =        FPR.test
  typeII.err.test         =        1 - TPR.test
  FPR_test[i]             =        typeI.err.test
  TPR_test[i]             =        1 - typeII.err.test
  print(paste('FPR for test set is',FPR_test[i]))
  print(paste('TPR for test set is',TPR_test[i]))
}

train = data.frame(FPR = FPR_train, TPR = TPR_train, Set = 'Train', Threshold = thrs_seq)
test  = data.frame(FPR = FPR_test,  TPR = TPR_test,  Set = 'Test',  Threshold = thrs_seq)
df <- rbind(train, test)

# Using colAUC method in catools library to get the AUC value
library(caTools)

# ROC for training set
AUC_train = colAUC(prob.train, y.train, plotROC = F)[1]

# ROC for test set
AUC_test = colAUC(prob.test, y.test, plotROC = F)[1]

# Plot the ROC curves
ggplot(data = df, aes(x=FPR, y = TPR, col = Set))+
  geom_line(show.legend = T)+
  labs(title = 'ROC Curves for Training and Test from Ridge Logistic Regression', x = 'False Positive Rate (FPR)', y = 'True Positive Rate (TPR)' ) +
  annotate(geom="text", 
           x=c(0.5,0.5), 
           y=c(0.4,0.5), 
           label=c(paste('AUC of Test: ',round(AUC_test, 3)), paste('AUC of Train: ',round(AUC_train, 3))), 
           color=c('red', 'blue'))+
  scale_color_manual(values=c('red', 'blue'))


## Thrs = 0.5
print( paste('Type 1 Error for Training Set:',df[df$Threshold == 0.5 & df$Set=='Train', 'FPR'] ))
print( paste('Type 2 Error for Training Set:',(1 - df[df$Threshold == 0.5 & df$Set=='Train', 'TPR'] )))
print( paste('Type 1 Error for Test Set:',df[df$Threshold == 0.5 & df$Set=='Test', 'FPR'] ))
print( paste('Type 2 Error for Test Set:',(1 - df[df$Threshold == 0.5 & df$Set=='Test', 'TPR'] )))




## Optimal error loss
df_train = df[df$Set == 'Train', ]
df_test = df[df$Set == 'Test', ]

df_train[which.min(abs(df_train$FPR - (1-df_train$TPR))), 'Threshold']
df_test [which.min(abs(df_test$FPR - (1-df_test$TPR))), 'Threshold']

## for train
df_train[which.min(abs(df_train$FPR - (1-df_train$TPR))), ]
## for test
df_test [which.min(abs(df_test$FPR - (1-df_test$TPR))), ]







##############################################################3

## Elastic Net

Enet = cv.glmnet(x = X.train, y=y.train, family = "binomial", alpha = 0.5, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc")

## in the case of AUC measure
## lasso$lambda[which.max(lasso$cvm)] is the same as lasso$lambda.min

enet_best = glmnet(x = X.train, y=y.train, lambda = Enet$lambda[which.max(Enet$cvm)], family = "binomial", alpha = 0.5, intercept = TRUE, standardize = FALSE)

beta0.hat = enet_best$a0
beta.hat = as.vector(enet_best$beta)

thrs                    =        0.5
prob.train              =        exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat  ))
y.hat.train             =        ifelse(prob.train > thrs, 1, 0) #table(y.hat.train, y.train)
FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
P.train                 =        sum(y.train==1) # total positives in the data
N.train                 =        sum(y.train==0) # total negatives in the data
FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
typeI.err.train         =        FPR.train
typeII.err.train        =        1 - TPR.train

print(paste( "train: err        = ", sprintf("%.2f" , mean(y.train != y.hat.train))))
print(paste( "train: typeI.err  = ", sprintf("%.2f" , typeI.err.train)))
print(paste( "train: typeII.err = ", sprintf("%.2f" , typeII.err.train)))


pi.hat                  =        sum(y.train)/n.train
y.hat.train.rand        =        rbinom(n.train, 1, pi.hat) # draws n.train independent binary r.v.s that are 1 with prob. pi.hat 
FP.train                =        sum(y.train[y.hat.train.rand ==1] == 0) # false positives = negatives in the data that were predicted as positive
TP.train                =        sum(y.hat.train.rand[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
P.train                 =        sum(y.train==1) # total positives in the data
N.train                 =        sum(y.train==0) # total negatives in the data

FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
typeI.err.train         =        FPR.train
typeII.err.train        =        1 - TPR.train
print(paste("---------------random predictor --------------------"))
print(paste( "train: err        = ", sprintf("%.2f" , mean(y.train != y.hat.train.rand))))
print(paste( "train: typeI.err  = ", sprintf("%.2f" , typeI.err.train)))
print(paste( "train: typeII.err = ", sprintf("%.2f" , typeII.err.train)))




prob.test               =        exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
y.hat.test              =        ifelse(prob.test > thrs,1,0) #table(y.hat.test, y.test)  
FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
P.test                  =        sum(y.test==1) # total positives in the data
N.test                  =        sum(y.test==0) # total negatives in the data
TN.test                 =        sum(y.hat.test[y.test==0] == 0)# negatives in the data that were predicted as negatives
FPR.test                =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
TPR.test                =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity = recall
typeI.err.test          =        FPR.test
typeII.err.test         =        1 - TPR.test
print(paste("--------------- --------------------"))
print(paste( "test: err        = ", sprintf("%.2f" , mean(y.test != y.hat.test))))
print(paste( "test: typeI.err  = ", sprintf("%.2f" , typeI.err.test)))
print(paste( "test: typeII.err = ", sprintf("%.2f" , typeII.err.test)))




str(train_data[[1]])

word_index                   =     dataset_imdb_word_index() #word_index is a named list mapping words to an integer index
reverse_word_index           =     names(word_index) # Reverses it, mapping integer indices to words
names(reverse_word_index)    =     word_index

review_index                 =     1
decoded_review <- sapply(train_data[[review_index]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
}) 
# Decodes the review. 
# Note that the indices are offset by 3 because 0, 1, and 2 are reserved indices 
# for “padding,” “start of sequence,” and “unknown.”


cat(decoded_review)

library(dplyr)
obh                    =       order(beta.hat) 
mw                     =       50
word.index.negatives   =       obh[1:mw]
word.index.positives   =       obh[(p-(mw-1)):p]


negative.Words         =       sapply(word.index.negatives, function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
cat(negative.Words)



positive.Words         =       sapply(word.index.positives, function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
cat(positive.Words)


# -- Top 5 Post and Neg words
cat(rev(tail(positive.Words,5)),sep = ", ")
cat(head(negative.Words,5),sep = ", ")


## ROC
thrs_seq   <- c(seq(0,1, by = 0.01))
FPR_train <- TPR_train <- FPR_test <- TPR_test <- rep(0, length(thrs_seq))

prob.train                =        exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat))
prob.test                 =        exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat))

for (i in 1:length(thrs_seq)){
  thrs                    = thrs_seq[i]
  print(paste('For the threshold sequence:',sprintf("%.2f" , thrs_seq[i])))
  
  # for training set
  y.hat.train             =        ifelse(prob.train > thrs, 1, 0) #table(y.hat.train, y.train)
  FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
  P.train                 =        sum(y.train==1) # total positives in the data
  N.train                 =        sum(y.train==0) # total negatives in the data
  FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
  TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity = power
  typeI.err.train         =        FPR.train
  typeII.err.train        =        1 - TPR.train
  FPR_train[i]            =        typeI.err.train
  TPR_train[i]            =        1 - typeII.err.train
  print(paste('FPR for training set is',FPR_train[i]))
  print(paste('TPR for training set is',TPR_train[i]))
  
  # for test set
  y.hat.test              =        ifelse(prob.test > thrs,1,0) #table(y.hat.test, y.test)  
  FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
  P.test                  =        sum(y.test==1) # total positives in the data
  N.test                  =        sum(y.test==0) # total negatives in the data
  TN.test                 =        sum(y.hat.test[y.test==0] == 0)# negatives in the data that were predicted as negatives
  FPR.test                =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
  TPR.test                =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity = recall
  typeI.err.test          =        FPR.test
  typeII.err.test         =        1 - TPR.test
  FPR_test[i]             =        typeI.err.test
  TPR_test[i]             =        1 - typeII.err.test
  print(paste('FPR for test set is',FPR_test[i]))
  print(paste('TPR for test set is',TPR_test[i]))
}

train = data.frame(FPR = FPR_train, TPR = TPR_train, Set = 'Train', Threshold = thrs_seq)
test  = data.frame(FPR = FPR_test,  TPR = TPR_test,  Set = 'Test',  Threshold = thrs_seq)
df <- rbind(train, test)

# Using colAUC method in catools library to get the AUC value
library(caTools)

# ROC for training set
AUC_train = colAUC(prob.train, y.train, plotROC = F)[1]

# ROC for test set
AUC_test = colAUC(prob.test, y.test, plotROC = F)[1]

# Plot the ROC curves
ggplot(data = df, aes(x=FPR, y = TPR, col = Set))+
  geom_line(show.legend = T)+
  labs(title = 'ROC Curves for Training and Test from Elastic Net Logistic Regression', x = 'False Positive Rate (FPR)', y = 'True Positive Rate (TPR)' ) +
  annotate(geom="text", 
           x=c(0.5,0.5), 
           y=c(0.4,0.5), 
           label=c(paste('AUC of Test: ',round(AUC_test, 3)), paste('AUC of Train: ',round(AUC_train, 3))), 
           color=c('red', 'blue'))+
  scale_color_manual(values=c('red', 'blue'))


## Thrs = 0.5
print( paste('Type 1 Error for Training Set:',df[df$Threshold == 0.5 & df$Set=='Train', 'FPR'] ))
print( paste('Type 2 Error for Training Set:',(1 - df[df$Threshold == 0.5 & df$Set=='Train', 'TPR'] )))
print( paste('Type 1 Error for Test Set:',df[df$Threshold == 0.5 & df$Set=='Test', 'FPR'] ))
print( paste('Type 2 Error for Test Set:',(1 - df[df$Threshold == 0.5 & df$Set=='Test', 'TPR'] )))




## Optimal error loss
df_train = df[df$Set == 'Train', ]
df_test = df[df$Set == 'Test', ]

df_train[which.min(abs(df_train$FPR - (1-df_train$TPR))), 'Threshold']
df_test [which.min(abs(df_test$FPR - (1-df_test$TPR))), 'Threshold']

## for train
df_train[which.min(abs(df_train$FPR - (1-df_train$TPR))), ]
## for test
df_test [which.min(abs(df_test$FPR - (1-df_test$TPR))), ]



## Prof's code:

#######################################################################################################
prob.train              =        exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat  ))
# (a.iii) roc curves
prob.test               =        exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
dt                      =        0.01
thta                    =        1-seq(0,1, by=dt)
thta.length             =        length(thta)
FPR.train               =        matrix(0, thta.length)
TPR.train               =        matrix(0, thta.length)
FPR.test                =        matrix(0, thta.length)
TPR.test                =        matrix(0, thta.length)

# The ROC curve is a popular graphic for simultaneously displaying the ROC curve two types of errors for all possible thresholds. 
# The name “ROC” is historic, and comes from communications theory. It is an acronym for receiver operating characteristics.
# ROC curves are useful for comparing different classifiers, since they take into account all possible thresholds.
# varying the classifier threshold changes its true positive and false positive rate. 

for (i in c(1:thta.length)){
  # calculate the FPR and TPR for train data 
  y.hat.train             =        ifelse(prob.train > thta[i], 1, 0) #table(y.hat.train, y.train)
  FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
  P.train                 =        sum(y.train==1) # total positives in the data
  N.train                 =        sum(y.train==0) # total negatives in the data
  FPR.train[i]            =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
  TPR.train[i]            =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
  
  # calculate the FPR and TPR for test data 
  y.hat.test              =        ifelse(prob.test > thta[i], 1, 0)
  FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
  P.test                  =        sum(y.test==1) # total positives in the data
  N.test                  =        sum(y.test==0) # total negatives in the data
  FPR.test[i]             =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
  TPR.test[i]             =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity    
  # print(paste("K=", K, " ki=",ki, ", K-fold CV=", Kfold.CV.err[i]))
}
#auc.train = auc(FPR.train, TPR.train)
auc.train     =       sum((TPR.train[1:(thta.length-1)] + 0.5 * diff(TPR.train)) * diff(FPR.train))
auc.test      =       sum((TPR.test[1:(thta.length-1)] + 0.5 * diff(TPR.test)) * diff(FPR.test))
print(paste("train AUC =",sprintf("%.2f", auc.train)))
print(paste("test AUC  =",sprintf("%.2f", auc.test)))



errs.train      =   as.data.frame(cbind(FPR.train, TPR.train))
errs.train      =   data.frame(x=errs.train$V1,y=errs.train$V2,type="Train")
errs.test       =   as.data.frame(cbind(FPR.test, TPR.test))
errs.test       =   data.frame(x=errs.test$V1,y=errs.test$V2,type="Test")
errs            =   rbind(errs.train, errs.test)

plt             = ggplot(errs) + geom_line(aes(x,y,color=type)) + labs(x="False positive rate", y="True positive rate") +   ggtitle("ROC curve",(sprintf("train AUC=%.4f,test AUC =%0.4f",auc.train,auc.test)))




#######################################################################################################
# (a.iv) type 1 and type 2 errors for theta=0.5
print(paste(" Train : for theta=0.5, type I error =",sprintf("%.2f", FPR.train[thta==0.5])))
print(paste(" Train: for theta=0.5, type II error =",sprintf("%.2f", 1-TPR.train[thta==0.5])))
#######################################################################################################
# (a.vi) type 1 (FPR) = type 2 (1-TPR) = ? errors for theta=?
ii        =    which.min(abs(FPR.train - (1-TPR.train)))
tht       =    thta[ii]
typeI     =    FPR.train[ii]
typeII    =    1-TPR.train[ii]
print(paste("train: theta=",sprintf("%.2f", tht), ", type I err=", sprintf("%.2f", typeI) , "and type II err=", sprintf("%.2f", typeII)))

typeI     =    FPR.test[ii]
typeII    =    1-TPR.test[ii]
print(paste("test: theta=",sprintf("%.2f", tht), ", type I err=", sprintf("%.2f", typeI) , "and type II err=", sprintf("%.2f", typeII)))


ii=51
typeI     =    FPR.test[ii]
typeII    =    1-TPR.test[ii]
tht       =    thta[ii]
print(paste("test: theta=",sprintf("%.2f", tht), ", type I err=", sprintf("%.2f", typeI) , "and type II err=", sprintf("%.2f", typeII)))









