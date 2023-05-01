import numpy as np #  we use numpy for the calcluation for the compact than the python list.
import matplotlib.pyplot as plt # for the  creative an impact for the plt 
import math # for the mathematical function

def read_data(path): # we connect the path  to read the  data  of the given csv  file.
    fun=open(path) # the varible fun is decalre   to open the path for the file to read
    tidee=[] #  tidee is the empty set for the program.
    for margines in fun.readlines(): # for the margines of the fun with the readlines.
        margine = margines.strip().split(',') #  we strip  and slpits the margines 
        information=[] # we make the information with the empty set.
        for z in margine: #  for the  in the margine  value 
            z=z.replace(')','').replace('(','').replace(' ','') #  z is replace 
            information.append(z) # we need to append  the element.
        tidee.append(information) #we append the tidee function 
    return tidee # we return the writeen tidee value .

def working(input):    # for the working input 
    bitikivachevalue = 1 / (1 + np.exp(-input)) # for the value that we get we then to a calculation.
    return bitikivachevalue # we return the output value of the calcluation.

def enchance(apple, bfile,swott_rate,rept,limitation):  # for the enchance we decalre the function.
    size = apple.shape[0] # sixe id decalred  with the apple.shape with index 0 .
    burden = limitation["burden"] # we declare weight  of the dataset. 
    favrd = limitation["favrd"] # we decalare the favrd for the  onsided info of the output.
    for c in range(rept):  # for the c in the range rept .
        san = working(np.dot(apple, burden) + favrd) #  we declare the san working  with the  apple and the burden  withe the favrd values. 
        deprivation = -1/size * np.sum(bfile * np.log(san)) + (1 - bfile) * np.log(1-san) # we deprivate the -1/size  with the nump and the calcluation.
        dW = 1/size * np.dot(apple.T, (san - bfile)) #  we then dW for the np.dot  apple .T with the bfile.
        db = 1/size * np.sum(san - bfile) # we dp =1/size with the np.sum for the calculation with the bfile.
        burden -= swott_rate * dW # the weight -   for the swott with multplie the dw
        favrd -= swott_rate * db  # we favrd the swott_rate  for  value to calcluate
    
    limitation["burden"] = burden #  the limitation variable with the  burden value .
    limitation["favrd"] = favrd # for the favrd of the data 
    return limitation # we  retune the limitation value

def train(aple, bfile, swott_rate,rept,limitation): # we train the data  with the rept and bfile  rept_limitaion.
    limitation_out = enchance(aple, bfile, swott_rate, rept ,limitation) #  WE out thr limit out value of the xset bfile and rept here
    return limitation_out  # return the  value from the  above  line then push them into limitation_out.

def plot_3D(APLE,bfile,limitation,n_iter):# we draw a 3d plot for the input values from the given data .
    z = lambda aple,BFILE: (-limitation['favrd']-limitation['burden'][0]*aple-limitation['burden'][1]*BFILE) / limitation['burden'][2] # we make points on the z axis 
    
    mesh=np.linspace(-1.5,1.5,80) #mesh where we define the size and shape of the values.
    aple,BFILE=np.meshgrid(mesh,mesh) # we make aple,bfile  values to draw the mesh 
    
    fig = plt.figure() #  fig  value is attend to hlp  us to draw a fig 
    ax = fig.add_subplot(projection='3d') # for the 3D plane to draw
    ax.plot_surface(aple,BFILE,z(aple,BFILE)) #  for the z axis .
    ax.plot3D(APLE[bfile==0,0],APLE[bfile==0,1],APLE[bfile==0,2],'xr') # we make the lines of the graph with the specific values and red color .
    ax.plot3D(APLE[bfile==1,0],APLE[bfile==1,1],APLE[bfile==1,2],'oy') # we add 3D with oy of yellow color.
    ax.view_init(60,30) # and the view with 60 and 30 size
    title='Logistic Regression Hyperplane with rept: ' + str(n_iter) # we title the graph as logistic regression hyperplane with rept 
    
    plt.title(title) # we make title  for the plot to show
    plt.show() #  show  the ploteed graph
 
def accuracy(bfile,bfile_pred): # we make the accuracy function  for the bfile and bfile_pred.
    right = 0 #  we initaize the value with the 0
    for c in range(len(bfile)): # for evry value in c range  where the length is defined.
        if bfile[c] == bfile_pred[c]: # if the bfile of c is == to the bfile_ored 
            right += 1 # we need to increment the  right value
    return right / float(len(bfile))    # then return the  right /float with len ( bfile )

def predictions(aple,limitation): # we predict the aple set and th eprediction
    z=np.dot(aple,limitation['burden'])+limitation['favrd']  #  z as dot is done with limitation with  burden as consideration.
    pred=[] # we make a empty set 
    for c in working(z): # for every ekement inc in working z .
        
        if c>=0.5:# if c is greater then 0.5
            pred.append(1) # we then pred with aooend ,
        else:
            pred.append(0)# or else the append is 0 
    pred=np.array(pred) # we then prede  variable  with the array pred ,
    return pred # we return the pred value 

def main():
    
    print('START Q3_AB\n') # start the program  and print the value .
    path=r'datasets/Q3_data.txt' #we read the path for the data.txt which we given
    info=read_data(path) # we fecth the data amd move it to info.
    info=np.array(info) # we make the info with the array
    #Changing dataypes
    APLE=info[:,:-1] # for the aple with the info with the index and  jump for the -1 from the back if the list.
    APLE=APLE.astype(np.float64) # for the aple we make the aple with the astyoe with the  float 64.
    bfile=info[:,-1] #  with the bfile  we pass  or go through the string with -1.
    bfile[bfile=='W']=0 # with the bfile we make with the 0.
    bfile[bfile=='M']=1  #will make the vlaue of the bfile to 1 .
    bfile=bfile.astype(np.float64) # for the bfile  np of file mke 64 float.
    #Standarization
    APLE=(APLE-APLE.mean(axis=0))/APLE.std(axis=0) # we add the data of the value  and make axis with o.
    
    #initialize parametlimitationers
    init_limitation = {} # a null value is given to init_minitation.
    init_limitation["burden"] = np.zeros(APLE.shape[1]) # we make the weight  of the limitation.
    init_limitation["favrd"] = 0 # with the favrd data in the init limitation.
    swott_rate=0.01 # with the swott_limit with 0.01.
    rept=[10,20,50,100,150] # we make the rept with the mention values.
    
    for n_iter in rept: # fro the n_iter in rept 
        limitation=train(APLE,bfile,0.01,n_iter,init_limitation) # we train the data then it is mentioned and add them to limitations
        bfile_pred = predictions(APLE, limitation) # for the value is moved to  bfile and the limitation.
        print('Accuracy with no of rept =',str(n_iter),'is ',str(accuracy(bfile, bfile_pred))) # we print the accuracy with the str  of the bfile.
        plot_3D(APLE,bfile,limitation,n_iter)# we plot the graph with the 3d.
    print('END Q3_AB\n') # as we started we need to w end the progran and we need to print the statement .
    
if __name__ == "__main__":
    main()

    




 
