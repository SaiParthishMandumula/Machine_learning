import numpy as np # we need numpy  as  np  lib  for the calcluation
import matplotlib.pyplot as plt # we need matplot lib  for vreative and static works  in python
import math #for the math calcluation.
import Q3_AB as logreg # we used Q3 AB link for the calcluation
import Q3_C as loov #we use Q3_c for the calcluation.

def main():
	#print('START Q3_C\n')
    print('START Q3_D\n') # we print the start  time of the program.
    path=r'datasets/Q3_data.txt' # we add the path for the dataset  to excuate. 
    info=logreg.read_data(path) # we read the file and get the information  from the dataseet.
    info=np.array(info) # we take info and make array for the information.
    #Changing dataypes
    APLE=info[:,[0,1]] # we take a variable APLE to acess the  index of the dataset.
    APLE=APLE.astype(np.float64) # we  used astype.
    bfile=info[:,-1] # we take -1 to the bfile   value.
    bfile[bfile=='W']=0 # we  make w 
    bfile[bfile=='M']=1 #and another as M.
    bfile=bfile.astype(np.float64) #we need a float 64.
    #Standarization
    APLE=(APLE-APLE.mean(axis=0))/APLE.std(axis=0) # we make a axis to 0.
    
    #initialize parameters
    init_limitation = {} # we make the  init limitation  as empty set.
    init_limitation["burden"] = np.zeros(APLE.shape[1]) #we make the limitation  with burden and the np is wqual with 0 
    init_limitation["favrd"] = 0 # we  add the limitation with favrd output
    swott_rate=0.01 # we make the rate for the swott value .
    
    print('Leave one out certify score with age column is ',loov.certify(APLE, bfile, init_limitation)) # we print the aple,bfile and inti_limitation.
    print('END Q3_D\n') # we print the end of the program.
if __name__ == "__main__": #  we  make a name   and declare a main for the program to run.
    main()