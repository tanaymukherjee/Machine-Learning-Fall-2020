rm(list = ls())    #delete objects
cat("\014")        #clear console

#devtools::install_github("rstudio/tensorflow")
#1
#devtools::install_github("rstudio/keras")

library(keras)  #https://keras.rstudio.com/
library(tensorflow)
#Keras is a high-level neural networks API developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

#Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). 
#Reviews have been preprocessed, and each review is encoded as a sequence of word indexes (integers). 
#For convenience, words are indexed by overall frequency in the dataset, so that for instance 
#the integer "3" encodes the 3rd most frequent word in the data. This allows for quick filtering 
#operations such as: "only consider the top 10,000 most common words, but eliminate the 
#top 20 most common words".
# https://blogs.rstudio.com/ai/posts/2017-12-07-text-classification-with-keras/
#Lists are the R objects which contain elements of different types like − numbers, strings, vectors and another list inside it. A list can also contain a matrix or a function as its elements. 
p                   =     2500
imdb                =     dataset_imdb(num_words = p, skip_top = 00) #, skip_top = 10
train_data          =     imdb$train$x
train_labels        =     imdb$train$y
test_data           =     imdb$test$x
test_labels         =     imdb$test$y

numberWords.train   =   max(sapply(train_data, max))
numberWords.test    =   max(sapply(test_data, max))



#c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

#The variables train_data and test_data are lists of reviews; each review is a list of
#word indices (encoding a sequence of words). train_labels and test_labels are
#lists of 0s and 1s, where 0 stands for negative and 1 stands for positive:

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

vectorize_sequences <- function(sequences, dimension = p) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}


X.train          =        vectorize_sequences(train_data)
X.test           =        vectorize_sequences(test_data)

#str(X.train[1,])
y.train          =        as.numeric(train_labels)
n.train          =        length(y.train)
y.test           =        as.numeric(test_labels)
n.test           =        length(y.test)


#fit                     =        glm(y.train ~ X.train, family = "binomial")
library(glmnet) 
fit                     =        glmnet(X.train, y.train, family = "binomial", lambda=0.0)

beta0.hat               =        fit$a0
beta.hat                =        as.vector(fit$beta)

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



print(paste("word most associated with positive reviews = ", reverse_word_index[[as.character((which.max(beta.hat)-3))]]))
print(paste("word most associated with negative reviews = ", reverse_word_index[[as.character((which.min(beta.hat)-3))]]))


# -- Part a and b
cat(rev(tail(positive.Words,10)),sep = ", ")
cat(head(negative.Words,10),sep = ", ")

# We can also get the result by changing the mw variable to 10 but
# since we have already done it in class simply calling the head and tail function will give us the result.

# -- Part c

review_index  <- which.min(abs(prob.train-0.5))
review_index
decoded_review <- sapply(train_data[[review_index]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
}) 
cat(decoded_review)

# -- Part d
# Most Positive
review_index  <- which.min(abs(prob.train-1))
review_index
decoded_review <- sapply(train_data[[review_index]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
}) 
cat(decoded_review)

# Most Negative
review_index  <- which.min(abs(prob.train-0))
review_index
decoded_review <- sapply(train_data[[review_index]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
}) 
cat(decoded_review)

# -- Part e
df <- as.data.frame(X.train %*% beta.hat + beta0.hat)
names(df)[1] <- "Score"
df <- df %>% mutate(Type = ifelse(Score > 0, "Positive", "Negative")) 

library(ggplot2)
ggplot(df, aes (Score, fill = Type)) +
  geom_histogram(alpha = 0.5, aes (y = ..density..), position = "identity", bins = 50) +
  labs(title  = "Overlapping histogram of postive and negative reviews", x = "Prob. Score = X*beta.hat + beta0.hat", y = "Density")


# -- Part f

# To a create a dummy sequence of thresholds (theta) = 0, 0.1, 0.2, 0.3, 0.4,....,1
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
  labs(title = 'ROC Curves for Training and Test from IMDB dataset', x = 'False Positive Rate (FPR)', y = 'True Positive Rate (TPR)' ) +
  annotate(geom="text", 
           x=c(0.5,0.5), 
           y=c(0.4,0.5), 
           label=c(paste('AUC of Test: ',round(AUC_test, 3)), paste('AUC of Train: ',round(AUC_train, 3))), 
           color=c('red', 'blue'))+
  scale_color_manual(values=c('red', 'blue'))
  
  
  
  
  
  