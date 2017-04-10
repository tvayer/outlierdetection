# Anomaly Detection Using Auto Encoder
# Source :https://github.com/sgrvinod/Anomaly-Detection-using-a-Deep-Learning-Auto-Encoder/blob/master/Anomaly_Detection_Faces.R


#Read in required packages
if(!require(pixmap)) install.packages("pixmap", dependencies=T)
library(pixmap)
if(!require(h2o)) install.packages("h2o", dependencies=T)
library(h2o)

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
  x=read.pnm(file=paste(path01, f, sep="\\"))
  pixvec=as.vector(t(x@grey))
  pixvec=as.integer(pixvec>mean(pixvec))
  train.vectors[[f]]=pixvec
}
for (f in test.names[1:length(test.faces)]){
  x=read.pnm(file=paste(path02, f, sep="\\"))
  pixvec=as.vector(t(x@grey))
  pixvec=as.integer(pixvec>mean(pixvec))
  test.vectors[[f]]=pixvec
}
for (f in test.names[(length(test.faces)+1):length(test.names)]){
  x=read.pnm(file=paste(pathoutlier, f, sep="\\"))
  pixvec=as.vector(t(x@grey))
  pixvec=as.integer(pixvec>mean(pixvec))
  test.vectors[[f]]=pixvec
}

#Create dataframes of pixel vectors
train<-as.data.frame(t(as.data.frame(train.vectors)))
test<-as.data.frame(t(as.data.frame(test.vectors)))


#Initialize h2o cluster
h2o.init(nthreads=-1, max_mem_size='4g')
h2o.clusterInfo()




