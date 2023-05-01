import numpy as np # for the mathematical implementation
import math # for the math calcluation
import matplotlib.pyplot as plt # to plot the graph 
import Q2_AB as util1 # to the program of Q2_AB

def main():
    
    print('START Q2_C\n') # it will start the q2
	#path_train=r'datasets/Q1_B_train.txt'
    path_train=r'datasets/Q1_B_train.txt'# this will link the dataset 
    info=util1.read_data(path_train) # with the read data it will store the value .
    info=info[info[:,0].argsort()] # this will argsort the graph with the info
    a=info[:,0] # with the variable a  it will slice the index with the 0
    a=a.reshape(a.shape[0],1) # a.rshape  slice 0,1
    b=info[:,1] # this will decalre the value of with the info index 1
    b.reshape(a.shape[0],1) # b.reshape the index  is 0 and 1
    
    path_test=r'datasets/Q1_C_test.txt' # we read the dataset  to test.
    info_test=util1.read_data(path_test) # this wit info test  of info from the dataset .
    
    a_test=info_test[:,0] # with the slicing o 
    a_test=a_test.reshape(a_test.shape[0],1) # a-test it will then reshape the sample test with the slice 0,1
    b_test=info_test[:,1] # btest  with the slice starting to 1 
    b_test.reshape(b_test.shape[0],1) # b_test it will reshape with the slicing 0 to 1 
    
    generator=0.204 # it will generator with the value 0.204
    
    b_pred=[] #an empty set for the b_pred 
    for point in a: #  for the point in a 
        b_pred.append(util1.non_global_coast_linear_regression(point[0], a, b,generator)[0,0])  #it will append the b_pred 
    a_pred=np.array(b_pred) # it will a_pred with the np
    
    b_pred_test=[] # then b pred test with the empty null
    for point in a_test: # for the point with a_test .
        b_pred_test.append(util1.non_global_coast_linear_regression(point[0], a, b,generator)[0,0])
    
    b_pred_test=np.array(b_pred_test) # b_trend  to np the array(b_pred _test)
    
    print('MAE on Train info is',np.mean(np.abs(b-b_pred))) # it will MAE the train the info is 
    print('MAE on test info is',np.mean(np.abs(b_test-b_pred_test))) # it will then b_pred_ test.
    print('END Q2_C\n')# it will print ot end .
if __name__ == "__main__":
    main()
    