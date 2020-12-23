rm(list = ls())    #delete objects
cat("\014")        #clear console
library(keras)  #https://keras.rstudio.com/
library(MESS) # calculate auc
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




index.to.review <- function(index) {
  decoded_review <- sapply(train_data[[index]], function(index) {
    word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
    if (!is.null(word)) word else "?"
  }) 
  return(decoded_review)
}

cat(index.to.review(10))
# Decodes the review. 
# Note that the indices are offset by 3 because 0, 1, and 2 are reserved indices 
# for “padding,” “start of sequence,” and “unknown.”


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

# (a) What are the top 10 words associated with positive reviews?
print(paste("word most associated with positive reviews = ", reverse_word_index[[as.character((which.max(beta.hat)-3))]]))

# (b) What are the top 10 words associated with negative reviews?
print(paste("word most associated with negative reviews = ", reverse_word_index[[as.character((which.min(beta.hat)-3))]]))

# (c) How can we identify the review in the training set that is hardest to classify? Find it and present the review. (1 point)
prob.train              =        exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat  ))


hardest.to.classify     =        which.min(abs(prob.train-0.5))
most.positive           =        which.max(prob.train)
most.negative           =        which.min(prob.train)
print(paste("-------------------------------------------------------------"))
print(paste("The hardest review to classify for this logisitc regression model is:  "))
cat(index.to.review(hardest.to.classify))
print(paste("-------------------------------------------------------------"))
print(paste("The most positive review based on this logisitc regression model is:  "))
cat(index.to.review(most.positive))
print(paste("-------------------------------------------------------------"))
print(paste("The most negative review based on this logisitc regression model is: "))
cat(index.to.review(most.negative))
print(paste("-------------------------------------------------------------"))


distance.P     =    (X.train[y.train==1, ] %*% beta.hat + beta0.hat) 
distance.N     =    X.train[y.train==0, ] %*% beta.hat + beta0.hat 

breakpoints = pretty( (min(c(distance.P,distance.N))-0.001):max(c(distance.P,distance.N)),n=200)
# n=200 above refers to the number of bins used in the histogram
# the large the number of bins, the higher the level of detail we see
hg.pos = hist(distance.P, breaks=breakpoints,plot=FALSE) # Save first
hg.neg = hist(distance.N, breaks=breakpoints,plot=FALSE) # Save second
color1 = rgb(0,0,230,max = 255, alpha = 80, names = "lt.blue")
color2 = rgb(255,0,0, max = 255, alpha = 80, names = "lt.pink")

library(latex2exp)

plot(hg.pos,col=color1,xlab=TeX('$x^T  \\beta +  \\beta_0$'),main =   paste("train: histogram  ")) # Plot 1st histogram using a transparent color
plot(hg.neg,col=color2,add=TRUE) # Add 2nd histogram using different color


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

print(paste("train AUC =",sprintf("%.4f", auc.train)))
print(paste("test AUC  =",sprintf("%.4f", auc.test)))


library(ggplot2)

errs.train      =   as.data.frame(cbind(FPR.train, TPR.train))
errs.train      =   data.frame(x=errs.train$V1,y=errs.train$V2,type="Train")
errs.test       =   as.data.frame(cbind(FPR.test, TPR.test))
errs.test       =   data.frame(x=errs.test$V1,y=errs.test$V2,type="Test")
errs            =   rbind(errs.train, errs.test)

ggplot(errs) + geom_line(aes(x,y,color=type)) + labs(x="False positive rate", y="True positive rate") +
  ggtitle("ROC curve",(sprintf("train AUC=%.4f,test AUC =%0.4f",auc.train,auc.test)))

