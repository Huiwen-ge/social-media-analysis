library(tm)
library(dplyr)
library(data.table)
library(tidytext)
library(caTools)
library(randomForest)
library(e1071)
library(pROC)
sReadLines <- function(fnam) {
  f <- file(fnam, "rb")
  res <- readLines(f)
  close(f)
  res
}
docs1<-read.csv(text = sReadLines('C:/Users/VE1/Documents/complaint1700.csv'),stringsAsFactors = FALSE)
docs1$y = 0
docs2 <- read.csv(text = sReadLines('C:/Users/VE1/Documents/noncomplaint1700.csv'),stringsAsFactors = FALSE)
docs2$y = 1

#data description
comment_neg <- docs1[,c(1,3)]
comment_pos <- docs2[,c(1,3)]
names(comment_neg) = c("doc_id", "text")
names(comment_pos) = c("doc_id", "text")
docs_neg <- Corpus( DataframeSource(comment_neg) )
docs_pos <- Corpus( DataframeSource(comment_pos) )
tdm.neg = TermDocumentMatrix(docs_neg, control=dtm.control)
tdm.pos = TermDocumentMatrix(docs_pos, control=dtm.control)
#negative
neg <- as.matrix(tdm.neg)
negv <- sort(rowSums(neg),decreasing=TRUE)
od <- data.frame(word = rownames(neg),freq=negv)
od$word = rownames(od)
set.seed(1234)
wordcloud(words = od$word, freq = od$freq, min.freq = 1,     
          max.words=200, random.order=FALSE, rot.per=0.3,
          colors="black")

#positive
pos <- as.matrix(tdm.pos)
posv <- sort(rowSums(pos),decreasing=TRUE)
pd <- data.frame(word = rownames(pos),freq=posv)
pd$word = rownames(pd)
set.seed(1234)
wordcloud(words = pd$word, freq = pd$freq, min.freq = 1,     
          max.words=200, random.order=FALSE, rot.per=0.3,
          colors="Orange")


df = rbind(docs1,docs2)
mtweets <- df[,c(1,3)]
names(mtweets) = c("doc_id", "text") # The 1st column must be named "doc_id" and contain a unique string identifier for each document. 
# The 2nd column must be named "text".
docs <- Corpus( DataframeSource(mtweets) )
dtm.control = list(tolower=T, removePunctuation=T, removeNumbers=T, stopwords=c(stopwords("english"),stopwords("spanish"), stopwords("portuguese")),stemming=T)
dtm.full = DocumentTermMatrix(docs, control=dtm.control)
dtm = removeSparseTerms(dtm.full,0.99)

#training dataset
as_tidy <- tidy(dtm)
bing <- get_sentiments("bing")
as_bing_words <- inner_join(as_tidy,bing,by = c("term"="word"))

X = as.data.frame(as.matrix(dtm))
for(i in c(1:nrow(as_bing_words))){
  if(as_bing_words[i,4] == 'positive'){
    X[as.character(as_bing_words[i,1]),which(names(X)==as.character(as_bing_words[i,2]))] <- 1
  }else{
    X[as.character(as_bing_words[i,1]),which(names(X)==as.character(as_bing_words[i,2]))]<- -1
  }
}

#dataset without true value
df_p = read.csv('C:/Users/VE1/Documents/temp.csv')
df_p = df_p[-1]
comments <- df_p[,c(1,3)]
names(comments) = c("doc_id", "text")
docs.p <- Corpus( DataframeSource(comments) )
dtm.p = DocumentTermMatrix(docs.p, control=dtm.control)
dtm.p = removeSparseTerms(dtm.p,0.99)

as_tidy.p <- tidy(dtm.p)
bing.p <- get_sentiments("bing")
as_bing_words.p <- inner_join(as_tidy.p,bing.p,by = c("term"="word"))

X.p = as.data.frame(as.matrix(dtm.p))
for(i in c(1:nrow(as_bing_words.p))){
  if(as_bing_words.p[i,4] == 'positive'){
    X.p[as.character(as_bing_words.p[i,1]),which(names(X.p)==as.character(as_bing_words.p[i,2]))] <- 1
  }else{
    X.p[as.character(as_bing_words.p[i,1]),which(names(X.p)==as.character(as_bing_words.p[i,2]))]<- -1
  }
}

#prepare datasets
data <- X
X.names = colnames(X)
X.p.names = colnames(X.p)
not_in_X.p = X.names[!(X.names %in% X.p.names)]
not_in_X = X.p.names[!(X.p.names%in% X.names)]


data[not_in_X] <- 0
data.p <- X.p
data.p[not_in_X.p]<-0
data$y = df$y

#train models
set.seed(1234)
split <- sample.split(data$y, SplitRatio = 0.75)
training_set <- subset(data, split == TRUE)
test_set <- subset(data, split == FALSE)

#rf
classifier <- randomForest(x = training_set[-206], 
y = training_set$y,
nTree = 100)
y_pred <- predict(classifier, newdata = test_set[-206])
pred.class <- as.numeric( y_pred>0.7 ) # try varying the threshold distance
table(pred.class, test_set$y)

#svm
svm.model <- svm(training_set$y ~ ., data = training_set[-206])
pred <- predict( svm.model, test_set[-206] )
pred.class <- as.numeric( pred>0.7 ) # try varying the threshold distance
table(pred.class, test_set$y)


#Bayesian
nb.model <- naiveBayes( training_set[-206], factor( training_set$y) ) # encode the response as a factor variable
pred.class <- predict( nb.model, test_set[-206] )
table( pred.class, test_set$y )

#train model
y_pred <- predict(classifier, data.p)
pred.class <- as.numeric(y_pred>0.7)
sum(y_pred<0.7)
df_p$pred = y_pred
df_p$pred[df_p$pred<0.7]=0
df_p$pred[df_p$pred>=0.7]=1

data_final = df_p[df_p$pred==1,]
data_final$bad <- grepl('broke|never|lost|annoying|ignore|complaint|bad|suck|cancel|updates|late|',tolower(data_final$tweet))
table(data_final$bad)
data_final = data_final[data_final$bad == FALSE,]
data_final = data_final[-5]
write.csv(data_final,'Huiwen_Ge.csv')
