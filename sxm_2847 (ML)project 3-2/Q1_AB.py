import numpy as np

extent = 0
mi_share = 2


class make_decision_tree():# for the class which make the decision tree to create.
    def __init__(Inuse, miShares=mi_share, max_depth=extent):
        Inuse.root = None # if the node has the null value.
# for stopping the condition statements.
        Inuse.miShares = miShares #if inuse.mishars then mishare is  declare.
        Inuse.max_depth = max_depth # if the function = max_depth value.
def calculate_Accuracy(E_Actual, E_Estimate, normalize=True): # it will calculate Accuracy of actual and estimated value ..
    Accuracy = [] #we  create variable for the null value .
    for i in range(len(E_Estimate)): # for every value in the range .
        if E_Estimate[i] == E_Actual[i]: #if Estimation is the actual value.
            Accuracy.append(1)
        else:
            Accuracy.append(0)

    return np.round(np.mean(Accuracy), decimals=2)


class St():
    def __init__(Inuse, Unknown_inde=None, Highestlimit=None, left=None, right=None, Information_Profit=None, Coast=None):
        #  it will constructor  for  the decision to split  into  left slave and right slave.
        # for contruction for the decision node
        Inuse.Unknown_inde = Unknown_inde
        Inuse.Highestlimit=Highestlimit
        Inuse.left = left # for the inuse to move to left
        Inuse.right = right #for the inuse node to move to right.
        Inuse.Information_Profit = Information_Profit # inuse we met the profit
        # for the leaf node to  build.
        Inuse.Coast= Coast # the present inuse value is equalize or initialize to the coast.


class make_decision_tree(): # for the decision tree to make.
    def __init__(Inuse, miShares=mi_share, max_depth=extent): # for the init where the extent.
        Inuse.root = None # if there is any  null value are present.
        #  for the stoping conditions.
        Inuse.miShares = miShares # for the splits.
        Inuse.max_depth = max_depth

    def build_tree(Autonoic, Infoset, curr_depth=0):
        #   for the recursion to run for the particular time.
        X, E = Infoset[:, :-1], Infoset[:, -1] # for the X,e there is inforest  and the step function is -1
        number_demos, num_Unknown = np.shape(X) # for the number_demos and the np_shapes  of the variable x

        # it will share  until the all the conditions are met and satisfied it  will go on .
        if number_demos >= Autonoic.miShares and curr_depth <= Autonoic.max_depth:
            #  it will find  the best possible way to share the data.
            Best_Shares = Autonoic.get_Best_Shares(
                Infoset, number_demos, num_Unknown)
            #  it will run the program and check if it is positive..
            if Best_Shares["Information_Profit"] > 0:
                #  it will make to move the recursion to the leftside of the desicion tree.
                left_subtree = Autonoic.build_tree(
                    Best_Shares["Infoset_left"], curr_depth+1)
                # it will shift the recursion to it right side ..
                right_subtree = Autonoic.build_tree(
                    Best_Shares["Infoset_right"], curr_depth+1)
                # it will hleps to know the decision node value of the given data.
                return St(Best_Shares["Unknown_inde"], Best_Shares["Highestlimit"],
                             left_subtree, right_subtree, Best_Shares["Information_Profit"])

        # when the data got split it will help the best share to split the data to its leftnode.
        leaf_Coast = Autonoic.calculate_leaf_Coast(E)
        # it will help the left node value to return..
        return St(Coast=leaf_Coast)

    def get_Best_Shares(Autonoic, Infoset, number_demos, num_Unknown):
        # we write this function to create for the best leftnode or thr right node.

        #  when the data is split we need to store the data value or the data infomation..
        Best_Shares = {}
        max_Information_Profit = -float("inf")

        # here it will loop over all the features.
        for Unknown_inde in range(num_Unknown):
            Unknown_Coasts = Infoset[:, Unknown_inde]
            possible_Highestlimits = np.unique(Unknown_Coasts)
            # it will create the loop for the future data ..
            for Highestlimit in possible_Highestlimits:
                # when the dataset it will need to get current split.
                Infoset_left, Infoset_right = Autonoic.split(
                    Infoset, Unknown_inde, Highestlimit)
                # we need to make sure that slaves are not null values.
                if len(Infoset_left) > 0 and len(Infoset_right) > 0:
                    E, left_E, right_E = Infoset[:, -
                                                 1], Infoset_left[:, -1], Infoset_right[:, -1]
                    # it need to compute the information 
                    curr_Information_Profit = Autonoic.information_Achieved(
                        E, left_E, right_E)
                    # if there is any future best split  for the given_dataset it will update the value.
                    if curr_Information_Profit > max_Information_Profit:
                        Best_Shares["Unknown_inde"] = Unknown_inde # unknown inde for the unknow value . 
                        Best_Shares["Highestlimit"] = Highestlimit 
                        Best_Shares["Infoset_left"] = Infoset_left
                        Best_Shares["Infoset_right"] = Infoset_right
                        Best_Shares["Information_Profit"] = curr_Information_Profit
                        max_Information_Profit = curr_Information_Profit

        # when the bestsplit is  done  it will return the best return value.
        return Best_Shares

    def split(Autonoic, Infoset, Unknown_inde, Highestlimit):
        #  the function that to  share the data.

        Infoset_left = np.array(
            [queue for queue in Infoset if queue[Unknown_inde] <= Highestlimit])
        Infoset_right = np.array(
            [queue for queue in Infoset if queue[Unknown_inde] > Highestlimit])
        return Infoset_left, Infoset_right

    def information_Achieved(Autonoic, Master, l_Slave, r_Slave):
        #  we need a function  to get achieved so that the automatic ,master, l_slave vlaue and r_slave value is done..

        Load_l = len(l_Slave) / len(Master)
        Load_r = len(r_Slave) / len(Master)
        Achieved = Autonoic.entropy(
            Master) - (Load_l*Autonoic.entropy(l_Slave) + Load_r*Autonoic.entropy(r_Slave))
        return Achieved

    def entropy(Autonoic, E):
        #  a function value is used  to compute the entropy.
        class_labels = np.unique(E)
        entropy = 0
        for clse in class_labels:
            p_clse = len(E[E == clse]) / len(E)
            entropy += -p_clse * np.log2(p_clse) # By using in-bilt log function will calculate entropy
        return entropy

    def calculate_leaf_Coast(Autonoic, E):
        #  in the decision tree we need a leaf node or slave node it will compute to create a leaf-node.

        E = list(E)
        return max(E, key=E.count)

    

    def fit(Autonoic, X, E):
        #   when the dataset is split we need  train the decision tree   get the accuracy  of the test and the train value.

        Infoset= np.concatenate((X, E), axis=1)
        Autonoic.root =Autonoic.build_tree(Infoset)

    def predictor_function(Autonoic, X):
        # for the  newdata to begin or compute we define a function.

        preditions = [Autonoic.make_prediction(x, Autonoic.root) for x in X]
        return preditions

    def make_prediction(Autonoic, x, tree):
        #  when there is a single datapoint x we need a function to do..

        if tree.Coast != None:
            return tree.Coast
        Unknown_val = x[tree.Unknown_inde]
        if Unknown_val <= tree.Highestlimit:
            return Autonoic.make_prediction(x, tree.left)
        else:
            return Autonoic.make_prediction(x, tree.right)

def file_reader(track):
    #Getting the file information
    informationcontain_file = open(track)
    channel = informationcontain_file.readline() 
    knowledge = []
    while channel:
        channel = channel.rstrip()
        knowledge.append(channel)
        channel = informationcontain_file.readline()
        Sol = []
    #intializing the traing data as empty array
    Trainng_data = []
    for queue in knowledge:
        u = ""
        for fg in queue:
            if (fg == '(' or fg == ')' or fg == ' '):
                continue
            else:
                u += fg
        d = u.split(',')
        Trainng_data.append(d)

    for queue in Trainng_data:
        mortal= []
        #Appending the data to the mortal variable
        mortal.append(float(queue[0]))
        mortal.append(float(queue[1]))
        mortal.append(int(queue[2]))
        
        mortal2 = []
        mortal2.append(mortal)
        [[mortal],]
        if (queue[3] == 'M'):
            mortal2.append([1])
        else:
            mortal2.append([0])

        Sol.append(mortal2)
    return Sol
def main():
    print('START Q1_AB\n')

    # Taking the dataset
    X_Trainng_data = []
    X_test = []
    E_Trainng_data = []
    E_test = []

    Trainng_data = file_reader('datasets/Q1_train.txt') # reading the train data
    test_knowledge = file_reader('datasets/Q1_test.txt') # reading the test data

    len_Trainng_data= len(Trainng_data)
    for i in Trainng_data:
        X_Trainng_data.append(i[0])
        E_Trainng_data.append(i[1])

    len_test = len(test_knowledge)
    for i in test_knowledge:
        X_test.append(i[0])
        E_test.append(i[1])

    mi_share = 3
    extent = 1

    for i in range(5):
        # classifer is called here.
        classifier = make_decision_tree(  # Calling the decision tree calssifier
            miShares=mi_share, max_depth=extent)
        classifier.fit(X_Trainng_data, E_Trainng_data)

        print("DEPTH = ", extent) # Printing the depth of it
        E_Trainng_data_Estimate = classifier.predictor_function(X_Trainng_data) # Estimating the traing data value
        Trainng_data_Accuracy = calculate_Accuracy(E_Trainng_data, E_Trainng_data_Estimate)

        E_test_Estimate = classifier.predictor_function(X_test) # Estimating the test data value
        test_Accuracy = calculate_Accuracy(E_test, E_test_Estimate)
        # Finding the accuracy of train and test data
        print("Accuracy  | Trainng_data = ", Trainng_data_Accuracy, 
              " | Test =  ", test_Accuracy)
        extent = extent + 1

    print('END Q1_AB\n')


if __name__ == "__main__":
    main()
    

