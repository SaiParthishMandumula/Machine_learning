import numpy as np #  we are premit to  use numpy for the compact and impact  in python 
import matplotlib.pyplot as plt # we use matplotlib  for creative and impact.
import math # for the mathematical calcluation
import Q3_AB as logreg # we connect the link to the QB

def certify(APLE,bfile,init_limitation,swott_rate=0.01,n_iter=50): # we define the function for the cerity with the limitation  with the function
    right_pred=0 # we made  right_pred for the  right function
    for c in range(APLE.shape[0]): # for the c value  with the APLE and the shape  
        row=APLE[c,:] # we declare the variable  row for the set
        real_label=bfile[c] # for the real_label we add it to the bfile  
        
        APLE_new=np.delete(APLE,c,axis=0) # we made APLE_new  to delete the of the another value with axis =0.
        bfile_new = np.delete(bfile,c) # we write the variable bfile_new for the del value of bfile with the elements of c to store  .
        row=row.reshape(1,APLE.shape[1]) # we made for the row and the shape and the  index [1] 
        
        limitation=logreg.train(APLE_new,bfile_new,0.01,n_iter,init_limitation) # we declare the variable  limitation with the function.
        predicted_label = logreg.predictions(row, limitation) #  we make the predicted value to the label.
        
        if real_label==predicted_label[0]: # if the real_label is equal to the  predicted _ label with the index 0.
            right_pred+=1 # we increment the right_ pred with one value . 
        
    return right_pred/APLE.shape[0] # we return the value  with the right_pred and the shape with the index value 0.

def main():  # we   decalre the main function.
    print('START Q3_C\n') #  we  print the statement with the Q3 problem.
    path=r'datasets/Q3_data.txt' # we nedd to  make the path for the dataset for the evaluate the function.
    info=logreg.read_data(path) # we make the info of the dataset  to read the file  to run the program.
    info=np.array(info) # we make the array   of the info present in the dataset.
    #Changing dataypes
    APLE=info[:,:-1] # we make the APLE function with the last index -1 value .
    APLE=APLE.astype(np.float64) # we make the  APLE   with the  float64.
    bfile=info[:,-1]   #we declare the bfile with the index last -1  with 3 
    bfile[bfile=='W']=0 # we make  the  bfile  with th W and equal to  the 0.
    bfile[bfile=='M']=1 # we make the bfile with the M and equal to 1.
    bfile=bfile.astype(np.float64) #  bfile as type with the float value 64.
    #Standarization
    APLE=(APLE-APLE.mean(axis=0))/APLE.std(axis=0) # we make APLE with the axis =0 and the std with the axis =0
    
    #initialize parameters
    init_limitation = {}  #   with the limit the parameter with the   empty set.
    init_limitation["burden"] = np.zeros(APLE.shape[1]) # we made the  parameter to  with the burden with the APle with the shape 1
    init_limitation["favrd"] = 0 #  with the limitation fvrd we intial the value with 0
    swott_rate=0.01 # the swott value is 0.01
    
    print('Leave one out certify score with age column is ',certify(APLE, bfile, init_limitation)) # we print the value with the  certify with the APLE .
    
	#print('END Q3_C\n')
    print('END Q3_C\n') # we print the  value with the end value .
    


if __name__ == "__main__": # we make the name function to the program.
    main()
    