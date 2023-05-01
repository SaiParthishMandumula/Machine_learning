import numpy as np # we declare the numoy  lib for the impact.
import math # math for  mathematics function
import matplotlib.pyplot as plt #   for the matplotlib for the graph

def read_data(path): # to read the path
    information_set = np.genfromtxt(path,delimiter=' ') #  we coonect the path to read the input file.
    information_set = information_set[:,[1,3]] #  we slice the set with the 1 to 3
    return information_set # we return the  information.

def coast(mark,A,genetor=0.204): #  the coast with maark A genetor with the 0.204 
    msp_set,nsp_set = np.shape(A) # with the msp_set and the Nsp_set  with the shap A
    coast=np.mat(np.eye((msp_set))) #  with the coast varible with the np.eye append 
    
    for k in range(msp_set): # for the k in range with the msp set
        variation=mark-A[k] # variation with the mark_A[k]
        coast[k,k]=np.exp(np.dot(variation,variation.T)/-(2*genetor**2)) # with the coast with k by k matrix and the variation with 2*genetor**2
    return coast #  return the coast 

def non_global_coast(mark,A,y_variable,genetor=0.204): #  we define with the global and coast 
    mass=coast(mark,A,genetor) # with the mass function then the mark A ,genetor .
    y_variable=y_variable.reshape(A.shape[0],1) #  for the variable y with the reshape 
    framework=(A.T*(mass*A)).I*(A.T*(mass*y_variable)) # for the framework then the I*  value.
    return framework # we return the frame work

def non_global_coast_linear_regression(mark,A,y_variable,genetor=0.204): #   we define the coast linear regression
    msp_set=A.shape[0] # with msp_set with the A.shape with index 0
    A = np.append(np.ones(msp_set).reshape(msp_set,1),A,axis=1) #   we declare the value A 
    mark=np.array([1,mark]) # we make mark  the array with the 1 index .
    framework=non_global_coast(mark, A, y_variable, genetor) # for the framework we decalre it with global coast with mark and A variable.
    predicted=np.dot(mark,framework) #  we predict the frame work with the np.dot 
    return predicted # we returnt  the predicted value

        

def main():
	
    print('START Q2_AB\n') # we print the start statement 
    path_train=r'datasets/Q1_B_train.txt' # we conncet the dataset to read 
    info=read_data(path_train) # we declare the info value with the read path 
    info=info[info[:,0].argsort()] # we move the argsort with the index with the argsort
    a=info[:,0] # with the a =info with  the slice 0
    a=a.reshape(a.shape[0],1) # we slice with the index 0 and 1
    y_variable=info[:,1] # y_variable with the info with the  slice 1
    y_variable.reshape(a.shape[0],1) # y_variable with the reshape  with the slice 0 and 1
    
    genetor=0.204 # we generator with the 0.204
    y_pred=[] # am empty set is declared 
    for mark in a: #  for the max in a 
        y_pred.append(non_global_coast_linear_regression(mark[0], a, y_variable,genetor)[0,0]) # when y_pred append with the non_global
    
    y_pred=np.array(y_pred) #y_pred with the np.array
    plt.scatter(a,y_variable, color='red',marker='^') # we make the color with the red and the mark with the ^
    plt.plot(a,y_pred,color='black') #   and the color with the black
    plt.title('non_global_coast_Linear Regression') #  with the non global with the coast linear regression.
    
    plt.xlabel('X') # we plt.xabel  to x
    plt.ylabel('Y') # we plt the value with y
    plt.show() # to show the plot 
    print('END Q2_AB\n') # we end the program here.
if __name__ == "__main__":
    main()
