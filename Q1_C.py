import math # we make for the mathematical calcluation
import numpy as np # for lib we use numoy compact and impact in than list  in the python.
import matplotlib.pyplot as plt # matplot lib  for the mathematical calcluation for the graph.
import Q1_AB as util1 #  we import the program Q1_AB.



def main(): #  we start main function
    print('START Q1_C\n') # we print start  program.
    traveller_train=r'datasets/Q1_B_train.txt' # we travel or conncet the value for the dataset for the value.
    info=util1.read_data(traveller_train)  #  info value is update with the  imout data file.
    info=info[info[:,0].argsort()] # for the info then we  make argsort of the data
    aple=info[:,0] # aple with the index and implement with the 0
    aple=aple.reshape(aple.shape[0],1) #  we make the shape with the index 0 and 1
    bfile=info[:,1] #  bfile with the info index 1.
    bfile.reshape(aple.shape[0],1) # bfile  with  reshape with the aple set with 0 and 1 index .
    
    travaller_test=r'datasets/Q1_C_test.txt' #   travaller_test  for the  test data for the path.
    info_test=util1.read_data(travaller_test) #  for the info test we  read data with the travller test.
    aple_test=info_test[:,0] # for the aple_test we declare the index of end with 0.
    aple_test=aple_test.reshape(aple_test.shape[0],1) #  reshape the aple with the 0 and 1
    y_test=info_test[:,1] #  for the y_test with the indx 1
    y_test.reshape(aple_test.shape[0],1) # for the  reshape then we take the value of the node with the 0 and  1 .
    
    S=[1,2,3,4,5,6,7,8,9,10] #  we make set with the given values.
    P=[1,2,3,4,5,6] # for the set with p value .
    min_error_test=100000 #  with the min_error test as 100000
    s_min=0 # we inital the value with the 0
    p_min=0 # we intial then the  0
    for s in S: # for the s in set s 
        for p in P: # for p in p set .
            
            
            A=util1.generate_input(aple, s, p) # A  with aple and s and p will generate 
            A_test=util1.generate_input(aple_test, s, p) # we then aple _test then s and p
            params=util1.linear_regression_calculation(A, bfile, s, p) #  with the params  with the a bfile  with s set and  set.
            
            prediction=util1.predict_ols(A, params) #  we then predict then until1 .
            prediction_test=util1.predict_ols(A_test, params) # we make the prediction with the until1 A test and params.	
            
            MAE_train=np.mean(np.abs(bfile-prediction)) # MAE_Train  then the bfile _prediction
            MAE_test=np.mean(np.abs(y_test-prediction_test)) # mae_test with the mean with y _test prediction.
            
            print('s=',str(s),' p=',str(p),' Error on train:',str(MAE_train),' Error on test:', str(MAE_test)) # we print  if there is a error then  test.
            
            if MAE_test<min_error_test: #  if MAEA_test with less than min error _test.
                min_error_test=MAE_test #  with the  min_error test then the MAE_test 
                s_min=s # with the s min then the s 
                p_min=p # with  the p min value in the p
    
    print('Minimum error on test data is for s=',str(s_min),' p=',str(p_min),' and error=',str(min_error_test)) # we print the  satetement and even we will have the3 error value.
    print('END Q1_C\n') #  it print the  end of the  program
if __name__ == "__main__": #  for the name with the main function.
    main()
    