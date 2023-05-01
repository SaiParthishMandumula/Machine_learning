import numpy as np # we use numpy lib for the impact then the list in python
import matplotlib.pyplot as plt #  for the mathematical calcluation
import math # for the f=math function.
def read_data(path): # we  find the path to read the file .
    given_info = np.genfromtxt(path,delimiter=' ') #  we take the info with the genfrom txt.
    given_info= given_info[:,[1,3]] # from the given info we make the index  with the 1,3
    return given_info # returnt he given info from the set.

def predict_ols(A,parameter): #  predict the values of the input given dataset.
    return np.dot(A,parameter) # then return the dot with the parameter 

def linear_regression_calculation(A,bfile,c,d): # we make the function with the linear regression  for the calcluation for the given dataset,
    parameter=np.dot(np.dot(np.linalg.inv(np.dot(np.array(A).transpose(), np.array(A))), A.transpose()), bfile) #  for the parameter with the dot  with np lib
    return parameter #  return the parameter.

def generate_input(aple,bfile,c): # generate the input value with the aple and bfile with c element 
    Y=np.ones((aple.shape[0],1)) # When y is np.ones then the shape with the index with the 0 and 1 index .
    for j in range(1,c+1): # for the j in the range with 1 ,c+1 
        sin = (np.sin(bfile*j*aple))**2 # sin with the np.sin then the bfile with **2.
        Y=np.hstack((Y,sin)) #  then Y np. with the hstack then y,sin.
    return Y # return the y .

def main():
	#print('START Q1_AB\n')
    print('START Q1_AB\n') #  it will print the start of the program 
    traveller=r'datasets/Q1_B_train.txt'# this  will  travel and connect the path to the dataset of the Q1_B train.txt.
    given_info=read_data(traveller) #  for the info of the file then the  read the data of the travller .
    given_info=given_info[given_info[:,0].argsort()] #  with the given info the value is index with the  value.
    aple=given_info[:,0] #  with the  aple the info from dataset with the 0
    aple=aple.reshape(aple.shape[0],1) # with the vale then reshape the index with the  0 and 1.
    bfile=given_info[:,1] # with the slice of the given info we go from start to end with 1 jump
    bfile.reshape(aple.shape[0],1) #  with the bfile we then reshpe the aple. shape with the index 0 and 1
     
    S=[1,2,3,4,5,6,7,8,9,10] #  with the s set we have the values
    P=[1,2,3,4,5,6] # we decalre some value with p set.
    for s in S: # for every set of s in s .
        for p in P: # for the set in p in the p set 
            A=generate_input(aple, s, p) # when A is generated the inout values aple and s and p
            parameter=linear_regression_calculation(A, bfile, s, p) # with the parameters then the linear regression with the calcli A,bfile and s adp
            prediction=predict_ols(A, parameter) # for the prediction_ols then A and the parameter  into predictoion.
            plt.scatter(aple,bfile,color='red',marker='^') # we plt the graph with the color red and shape with ^.
            plt.plot(aple,prediction,color='blue') #  this will plot the  aple,prediction with the color = blue.
            plt.title('Linear Regression with sin basis function k='+str(s)+' and d='+str(p)) #  thi is to tittle  of the graph side with the linear regresiion.
            plt.xlabel('Independent variable') # ti label the value with x
            plt.ylabel('Dependent variable',rotation=90) # to labely axis with the rotation 90
            plt.show() # this is show the plot graph of the program .
    
    print('END Q1_AB\n') # print  statement after the end of the task
if __name__ == "__main__":
    main()