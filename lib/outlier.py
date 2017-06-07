import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import random
import math
import time
import scipy.spatial.distance as distance
from sklearn.covariance import MinCovDet as MCD
import scipy.stats as stats
from numpy import linalg as LA
import progressbar
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator,TransformerMixin


class OutlierMahalanobis(TransformerMixin):

    def __init__(self, support_fraction = 0.95, verbose = False, chi2_percentile = 0.995,qqplot=True):
        self.verbose = verbose
        self.support_fraction = support_fraction
        self.chi2 = stats.chi2
        self.mcd = MCD(store_precision = True, support_fraction = support_fraction)
        self.chi2_percentile = chi2_percentile
        self.qqplot=qqplot

    def get_params(self):
        return {"support_fraction": self.support_fraction,"chi2_percentile":self.chi2_percentile}

    def set_params(self, **parameters):
        for key,value in parameters.items() :
            setattr(self,key,parameters[key])
        return self

    def fit(self, X,y=None):
        """Prints some summary stats (if verbose is one) and returns the indices of what it consider to be extreme"""
        self.mcd.fit(X)
        d = np.array([distance.mahalanobis(p, self.mcd.location_, self.mcd.precision_ ) for p in X])
        self.d2 = d**2 #MD squared
        n, self.degrees_of_freedom_ = X.shape
        self.iextreme_values = (self.d2 > self.chi2.ppf(self.chi2_percentile, self.degrees_of_freedom_) )
        if self.verbose:
            print("%.3f proportion of outliers at %.3f%% chi2 percentile, "%(self.iextreme_values.sum()/float(n), self.chi2_percentile))
            print("with support fraction %.2f."%self.support_fraction)
            pvalue=stats.kstest(self.d2, lambda x : stats.chi2.cdf(x,df=self.degrees_of_freedom_))[1]
            if pvalue <= 0.01:
                print('Attention : Très forte présomption contre l\'hypothèse nulle p_value : '+str(pvalue))
            elif pvalue <= 0.05:
                print('Attention : Forte présomption contre l\'hypothèse nulle p_value : '+str(pvalue))
            elif pvalue <= 0.1:
                print('Faible présomption contre l\'hypothèse nulle p_value : '+str(pvalue))
            else :
                print('Pas de présomption contre l\'hypothèse nulle. p_value : '+str(pvalue))
            if self.qqplot==True :
                plt.figure(figsize=(10,10))
                stats.probplot(self.d2,dist=stats.chi2(df=self.degrees_of_freedom_), plot=plt)
                plt.title('QQ plot between Mahanalobis distance quantiles and Chi2 quantiles')
                plt.show()

        return self

    def transform(self,X):

        return X[~self.iextreme_values]

    def plot(self,log=False, sort = False ):
        """
        Cause plotting is always fun.

        log: transform the distance-sq to a log ( distance-sq )
        sort: sort the data according to distnace before plotting
        ifollow: a set if indices to mark with yellow, useful for seeing where data lies across views.

        """
        n = self.d2.shape[0]
        fig = plt.figure(figsize=(10,10))

        x = np.arange( n )
        ax = fig.add_subplot(111)


        transform = (lambda x: x ) if not log else (lambda x: np.log(x))
        chi_line = self.chi2.ppf(self.chi2_percentile, self.degrees_of_freedom_)

        chi_line = transform( chi_line )
        d2 = transform( self.d2 )
        if sort:
            isort = np.argsort( d2 )
            ax.scatter(x, d2[isort], alpha = 0.7, facecolors='none' )
            plt.plot( x, transform(self.chi2.ppf( np.linspace(0,1,n),self.degrees_of_freedom_ )), c="r", label="distribution assuming normal" )


        else:
            ax.scatter(x, d2 )
            extreme_values = d2[ self.iextreme_values ]
            ax.scatter( x[self.iextreme_values], extreme_values, color="r" )

        ax.hlines( chi_line, 0, n,
                        label ="%.1f%% $\chi^2$ quantile"%(100*self.chi2_percentile), linestyles = "dotted" )

        ax.legend()
        ax.set_ylabel("distance squared")
        ax.set_xlabel("observation")
        ax.set_xlim(0, self.d2.shape[0])


        plt.show()

class R_pca(TransformerMixin):

    def __init__(self, mu=None, lmbda=None,tol=None,max_iter=1000,iter_print=100,nb_extreme=100):
        self.tol=tol
        self.max_iter=max_iter
        self.iter_print=iter_print
        self.mu=mu
        self.lmbda=lmbda
        self.nb_extreme=nb_extreme

    def get_params(self):
        return {"mu": self.mu,"lmbda":self.lmbda,"tol":self.tol,"max_iter":self.max_iter}

    def set_params(self, **parameters):
        for key,value in parameters.items() :
            setattr(self,key,parameters[key])
        return self


    @staticmethod
    def norm_p(M, p):
        return np.sum(np.power(M, p))

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self,X,y=None):
        self.S = np.zeros(X.shape)
        self.Y = np.zeros(X.shape)

        if self.mu is None:
            self.mu = np.prod(X.shape) / (4 * self.norm_p(X, 2))

        self.mu_inv = 1 / self.mu

        if self.lmbda is None:
            self.lmbda = 1 / np.sqrt(np.max(X.shape))


        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(X.shape)

        if self.tol:
            _tol = self.tol
        else:
            _tol = 1E-7 * self.norm_p(np.abs(X), 2)

        while (err > _tol) and iter < self.max_iter:
            Lk = self.svd_threshold(
                X - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                X - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (X - Lk - Sk)
            err = self.norm_p(np.abs(X - Lk - Sk), 2)
            iter += 1
            if (iter % self.iter_print) == 0 or iter == 1 or iter > self.max_iter or err <= _tol:
                print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        norm=LA.norm(self.S,axis=0)
        self.iextreme_values=np.argsort(norm)[::-1][0:self.nb_extreme]

        return self

    def transform(self,X):
        indices=np.in1d(np.array(range(len(X))),self.iextreme_values)
        return X[~indices]

    def plot_normC(self):
        norm=np.sort(LA.norm(self.S,axis=0))[::-1]
        plt.figure(figsize=(10,10))
        plt.plot(norm,'r-')
        plt.legend(['Norme 2 de $C_{0}$'],fontsize=15)
        plt.title('Norme 2 de $C_{0}$ pour chaque point',fontsize=15)
        plt.show()

class OutliersKmeans(TransformerMixin):
    ''' v.0.2 OutliersKmeans : Find outliers using Kmeans. Only for semi-supervised outlier detection'''
    def __init__(self,kmeans,parallel=False,showbar=False):
        self.kmeans=kmeans
        self.parallel=parallel
        self.showbar=showbar

    def get_params(self):
        return {"kmeans": self.kmeans}

    def set_params(self, **parameters):
        for key,value in parameters.items() :
            setattr(self,key,parameters[key])
        return self


    def fit(self,X,y=None):

        self.kmeans.fit(X)
        preds=self.kmeans.predict(X)
        self.centers=self.kmeans.cluster_centers_
        self.d=self.find_extreme_point(self.centers,preds,X)

        return self

    def transform(self,X):

        self.iextreme_values=[]
        if self.showbar == True :
            widgets = [progressbar.Percentage(), progressbar.Bar()]
            bar = progressbar.ProgressBar(widgets=widgets, max_value=len(X)).start()
            j=0
        if self.parallel==True:
            p=Pool(4)
            g=lambda point:not(self.in_any_circle(point,self.centers,self.d))
            self.iextreme_values=p.map(g, X)

        else:
            for point in X:

                self.iextreme_values.append(not(self.in_any_circle(point,self.centers,self.d)))

                if self.showbar== True :
                    self.sleep(0.1)
                    bar.update(j)
                    j=j+1

        if self.showbar== True :
            bar.finish()

        self.iextreme_values=np.array(self.iextreme_values)
        return X[~self.iextreme_values]


    def sleep(self,delay):
        time.sleep(delay)


    def to_center_distances(self,X,mu):
        D=[]
        for i in range(len(X)):
            D.append(np.sum((X[i]-mu)**2))
        return D

    def find_extreme_point(self,centers,ypred,X):
        K=len(centers)
        d={}
        for i in range(K):
            X_in_cluster=X[np.where(ypred==i),:][0]
            D=self.to_center_distances(X_in_cluster,centers[i])
            indexmax=np.argmax(D)
            d[i+1]=(X[indexmax],math.sqrt(np.max(D)))
        return d

    def draw_circle(self,Npoints,radius,center):
        circle_slice=2*math.pi/Npoints
        p=[]
        for i in range(Npoints):
            angle=circle_slice*i
            newx=center[0]+radius*math.cos(angle)
            newy=center[1]+radius*math.sin(angle)
            p.append([newx,newy])
        return np.array(p)

    def bounds(self,centers,ypred,Xreduced,Npoints=100):
        d=self.find_extreme_point(centers,ypred,Xreduced)
        K=len(d)
        circles=[]
        for i in range(K):
            center=centers[i]
            radius=d[i+1][1]
            circles.append(self.draw_circle(Npoints,radius,center))
        return circles

    def in_circle(self,point,center,radius):
        return math.sqrt(np.sum((point-center)**2))<=radius

    def in_any_circle(self,point,centers,d):
        K=len(d)
        incirclek=[]
        for k in range(K):
            center=centers[k]
            radius=d[k+1][1]
            incirclek.append(self.in_circle(point,center,radius))
        return np.any(incirclek)
