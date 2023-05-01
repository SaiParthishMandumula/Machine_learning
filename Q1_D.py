import math #  decalre the  math function for the math matical calculation
import numpy as np #  we use numpy  for the  calcluation.
import matplotlib.pyplot as plt # for the matplot lib  to plt the graph.
import Q1_AB as util1 # we import the Q1_Ab file 

def main():
	#print('START Q1_C\n')
    print('START Q1_D\n') # it print the start q1_d .
    traveller_train=r'datasets/Q1_B_train.txt' #  we need to travel  for the path to access the data .
    info=util1.read_data(traveller_train) # for the info variable we input the data from the dataset.
    info=info[info[:,0].argsort()] #  taken index with 0 and argsort .
    info=info[:20,:] # we access the info and it will travel upto 20 index.
    aple=info[:,0] # we start the travel with the values with the 0
    aple=aple.reshape(aple.shape[0],1) #  it  will reshape the value of the aple .
    bfile=info[:,1] # for the bfile with the index and the move 1 segment.
    bfile.reshape(aple.shape[0],1) # for the bfile then the aple shape then  access the list from the index 0.
    traveller_test=r'datasets/Q1_B_train.txt' # for the traaveller test we need to  train the data for the iplementation.
    data_test= util1.read_data(traveller_test) # we make the set for the travller _test.
    aple_test=data_test[:,0] # for the index 0
    aple_test=aple_test.reshape(aple_test.shape[0],1) #  then make the shape of 0 index with the  1
    bfile_test=data_test[:,1] # then the bfile with the index with start qnd ending with 1
    bfile_test.reshape(aple_test.shape[0],1) #  reshape the bfile  with the index 0
    
    Setts=[1,2,3,4,5,6,7,8,9,10] # we make the set with the given number.
    P_set=[1,2,3,4,5,6] # p set with the number 
    min_error_test=100000 # we have the minmiom  test error 
    s_min=0 # make the min value form s set.
    p_min=0 # make the pmin value 0
    for s in Setts: #  for every s element in the setts 
        for p in P_set: #  for the every element p  in the pset 
            A=util1.generate_input(aple, s, p) #  we make a to which we generate the aple and the s p 
            A_test=util1.generate_input(aple_test, s, p) # a test  for which the unti1 generate the values with the s ,p
            params=util1.linear_regression_calculation(A, bfile, s, p) # foe the params we make the linear regression and then we calcluate the value.
            
            prediction=util1.predict_ols(A, params) #we make the prediction  with A ,params..
            prediction_test=util1.predict_ols(A_test, params) # we make the predict_test then A test with  params.
            
            plt.scatter(aple,bfile,color='red',marker='^') # for the indication  color is red then with the mark o as point.
            plt.plot(aple,prediction,color='black') # for the plot we make the value we give black.
            plt.title('Linear Regression with sin basis function s='+str(s)+' and p='+str(p)) #  we then title the plot .
            plt.xlabel('Independent variable') # we make the x label  with the independent variable.
            plt.ylabel('Dependent variable',rotation=90) #  for the y label we make deopendent variable and then rotation 90.
            plt.show() # we  then plot  to show .
            
            MAE_train=np.mean(np.abs(bfile-prediction)) # we then MAE_train 
            MAE_test=np.mean(np.abs(bfile_test-prediction_test)) #we make the y  test prediction
            
            print('s=',str(s),' p=',str(p),' Error on train:',str(MAE_train),' Error on test:', str(MAE_test)) #  print the value  and if there is error we print it .
            
            if MAE_test<min_error_test: # if  mae_test is less than min-error test
                min_error_test=MAE_test # min_error test = mae_test
                s_min=s # we make  min vallue from the s set
                p_min=p # we make the p min value for the p set.
    
    print('Minimum error on test data is for s=',str(s_min),' p=',str(p_min),' and error=',str(min_error_test))
    print('END Q1_D\n') #   it  will print the end statement .


if __name__ == "__main__":
    main()