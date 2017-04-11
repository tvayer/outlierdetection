# Anomaly Detection Using Auto Encoder 
# Source :https://github.com/sgrvinod/Anomaly-Detection-using-a-Deep-Learning-Auto-Encoder/blob/master/Anomaly_Detection_Faces.R

library(autoencoder)
library(pixmap)
path01='D:\\Users\\S37283\\Documents\\Outlier detection\\CroppedYale\\yaleB01'
path02='D:\\Users\\S37283\\Documents\\Outlier detection\\CroppedYale\\yaleB02'
pathoutlier='D:\\Users\\S37283\\Documents\\Outlier detection\\CroppedYaleNoisy\\yaleB02noisy'
train.names<-list.files(path=path01)
test.faces<-list.files(path=path02)

test.notfaces<-sample(list.files(path=pathoutlier),
                      length(test.faces),
                      replace=F)
test.names<-append(test.faces, test.notfaces)

#Get pixel vectors for these files
train.vectors<-list()
test.vectors<-list()
for (f in train.names){
  x=read.pnm(file=paste(path01, f, sep="\\"),nrow=10,ncol=12)
  pixvec=as.vector(t(x@grey))
  #pixvec=as.integer(pixvec>mean(pixvec))
  train.vectors[[f]]=pixvec
}
for (f in test.names[1:length(test.faces)]){
  x=read.pnm(file=paste(path02, f, sep="\\"))
  pixvec=as.vector(t(x@grey))
  #pixvec=as.integer(pixvec>mean(pixvec))
  test.vectors[[f]]=pixvec
}
for (f in test.names[(length(test.faces)+1):length(test.names)]){
  x=read.pnm(file=paste(pathoutlier, f, sep="\\"))
  pixvec=as.vector(t(x@grey))
  #pixvec=as.integer(pixvec>mean(pixvec))
  test.vectors[[f]]=pixvec
}

#Create dataframes of pixel vectors
train<-as.data.frame(t(as.data.frame(train.vectors)))
test<-as.data.frame(t(as.data.frame(test.vectors)))

# Autoencoing

N.hidden = 10 ## number of units in the hidden layer
lambda = 0.0002 ## weight decay parameter
beta = 6 ## weight of sparsity penalty term
rho = 0.01 ## desired sparsity parameter
epsilon <- 0.001
nn<-autoencode(as.matrix(train),
               N.hidden = N.hidden,
               epsilon = epsilon,
               lambda = lambda,
               rho = rho,
               beta = beta)
