import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as mpl
from numpy import genfromtxt
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import types


def pca(df):
	df=df.transpose()
	label=list(df.ix[:,(len(df.columns)-1)])
	matrix=StandardScaler().fit_transform(df.ix[:,:(len(df.columns)-1)])

	covariance=np.cov(matrix.T)
	eigen_val,eigen_vec=np.linalg.eig(covariance)

	eigens=list()

	for i in range(len(eigen_val)):
		eigens.append([(np.abs(eigen_val[i])),eigen_vec[:,i]])
	eigens.sort
	eigens.reverse
	eigen_total=sum(eigen_val)

	lam=[]
	cum_sum=0
	for value in eigen_val:
		cum_sum+=value
		lam.append(cum_sum/eigen_total)

	plt.plot(lam,marker='o')
	plt.xlabel("# of Features")
	plt.ylabel("Cumulative sum of eigen values/eigen value total")

	plt.show()
	last=[]
	name=[]
	for i in range(150):
		last.append(eigens[i][1].reshape(784,1))
		name.append(str(i))
	name.append("label")
	reduced=np.hstack(last)
	print matrix.shape
	print reduced.shape
	final=matrix.dot(reduced)
	final=np.real(final)
	df_out=pd.DataFrame(columns=name)
	for i in range(150):
		df_out.ix[:,i]= final[:,i]
	df_out['label']=label
	return df_out,reduced,name

def pca_test(df,model,name):
	df=df.transpose()
	label=list(df.ix[:,(len(df.columns)-1)])
	matrix=StandardScaler().fit_transform(df.ix[:,:(len(df.columns)-1)])
	final=matrix.dot(model)
	final=np.real(final)
	df_out=pd.DataFrame(columns=name)
	for i in range(150):
		df_out.ix[:,i]= final[:,i]
	df_out['label']=label

	return df_out

if __name__=='__main__':
	train=pd.read_csv('train.csv')
	df_train,model,name=pca(train)
	df_train.to_csv("pca_out.csv", sep=",")
	test=pd.read_csv('test.csv')
	df_test=pca_test(test,model,name)
	df_test.to_csv("pca_test_out.csv", sep=",")
# fig = plt.figure(figsize=(8,8))
# ax=mpl(fig,elev=150,azim=110)
# ax.scatter(final[:, 0],final[:, 1],final[:,2],c=label)
# plt.show()

