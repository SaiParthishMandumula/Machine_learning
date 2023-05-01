import numpy as np # we implement this numpy with impact   than the list
import math # for the mathematical  function 
import matplotlib.pyplot as plt # for the matplotlib  for the graph plot 
import Q2_AB as util1 # we enroute  Q2_AB with the implementation

def main():
    print('START Q2_D\n')# to start the porgram
	#path_train=r'datasets/Q1_B_train.txt'
    path_train=r'datasets/Q1_B_train.txt' #   we attacth the train.txt for the path 
    info = util1.read_data(path_train) #   with the variable info we path the  given data
    info = info[:20,:] # with the info we slice the element 
    info = info[info[:,0].argsort()] # with the info as variable we slice the argsort of the info then index with element 0
    a=info[:,0] # with built with a info then the sice with the 0.
    a=a.reshape(a.shape[0],1) # with a and the shape of the slice with 0 and 1 .
    b=info[:,1] # then the varible b with the slice with the index and ending 1.
    b.reshape(a.shape[0],1) # b,reshape with   with the index o and 1.
    
    path_test=r'datasets/Q1_C_test.txt' # it connect the  path to the test  file.
    info_test=util1.read_data(path_test) #  it will then make info with the read the data with the path .
    
    a_test=info_test[:,0] # we then make a_test  with the the index startinf and ending with the 0  jump.
    a_test=a_test.reshape(a_test.shape[0],1) #  a_test. reshape with the indesx with the o and 1 .
    b_test=info_test[:,1] # btest with the  sliceing with the 1 
    b_test.reshape(a_test.shape[0],1) # b_test with the reshape 
    
    generator=0.204 # generator with the value 0.204
    
    b_pred=[]# with the b_pred with the null value.
    for point in a: #  for the point in a 
        b_pred.append(util1.non_global_coast_linear_regression(point[0], a, b,generator)[0,0]) # b_pread with need to append  with the linear regression.
    b_pred=np.array(b_pred)# with the b_pred  with the np.
    
    b_pred_test=[]#  we declare the value with the null test.
    for point in a_test: # for the point in a_test
        b_pred_test.append(util1.non_global_coast_linear_regression(point[0], a, b,generator)[0,0]) # b_pred need to append with the linear regression with the a ,b generator.
    b_pred_test=np.array(b_pred_test) # b_pred_test with the array and the b_pread _test.
    
    print('MAE on Train info is',np.mean(np.abs(b-b_pred))) # with the print the mean then b-bA apend 
    print('MAE on test info is',np.mean(np.abs(b_test-b_pred_test))) #  it print the MAE with the test on  info with the b test and b_prend test.
    
    plt.scatter(a,b, color='red',marker='^') #  with the color red and the plot the points as ^ 
    plt.plot(a,b_pred,color='black') # we make the color black for the linear line
    plt.title('Locally Weighted Linear Regression') #  we declare the little as the local weight  with the liner regression.
    
    plt.xlabel('X') # with the plot label it as x 
    plt.ylabel('Y') # with the plot we label it as y
    plt.show() # we will show the graph 
    print('END Q2_D\n') # we then end the program .


if __name__ == "__main__":
    main()
    