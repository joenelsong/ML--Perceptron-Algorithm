""" 

# Author: Joey M Nelson
# 2/3/2016
# Runtime Environment: Python 3.5


Machine Learning >> Binary Classification >> Perceptron Algorithm 

"""
import sys
import csv
import random
import math

class Perceptron(object):
    
    MAX_ITER = 100
    LEARNING_RATE = 0.5
    THETA = 0 # Used to predict classification if threshold of calculated output is > Theta
    
    total_error = 0
    convergance = 0


    def __init__(self, data, randomize_weights = 1):
        self._rows = len(data)
        self._cols = len(data[0])
        self._labels = data[0]
        
        self._data = data[1:self._rows] # Subtract header row from data
        
        # Create random weights if flag is set
        self._weights = [0]*self._cols
        if ( randomize_weights == 1 ):
            for i in range(self._cols -1 ):
                randomValue = random.uniform( -1.0, 1.0)
                self._weights[i] = randomValue # 1 more for the bias this goes where the classifier would be
    
            self._weights[i+1] = 0 # initialize Bias to 0
        
        self._TestData = [[]] # Create empty TestData data structure
        
        
    def CalculateOutput(self, inputs):
        ''' This calculates the activation -1 or 1
            example: calculateOutput(inputs, weights):
            inputs = list of numerical/binary data; weights = list of weights for each data input
        '''
        #print("length of inputs: ", len(inputs))
        if len(inputs) != len(self._weights): # Error checking
            raise ValueError('Input and weight list size mistmatch!: input len = ', len(inputs), "weight size = ", len(self._weights) )
            
        sum = 0.0
         
        for i in range ( len(inputs) -1 ): # Minus 1 so we don't count the classification label
            sum += ( inputs[i] * self._weights[i] )

        sum += 1* self._weights[i+1] # add the bias
        #print(i)
        
        print (sum)

        return ( 1 if (sum > self.THETA ) else -1 )
        
        
    def Train(self):
        ''' Trains model based on training data set
        '''
        outcalc = 0
        iteration = 0
        while (iteration < self.MAX_ITER):
            self.total_error = 0
            for i in range( len(self._data) ): # Loop through each data record
                
                outcalc = self.CalculateOutput(self._data[i])
                err = self._data[i][self._cols-1] - outcalc

                # Update Weights
                for u in range(self._cols -1):
                    self._weights[u] += self.LEARNING_RATE * err * self._data[i][u]
                self._weights[u+1] += self.LEARNING_RATE * err * 1 # Add bias
                
                self.total_error += math.fabs(err); ## convergence count
                
            if ( self.total_error == 0 ):
                print("Training completed early due to convergence")
                break
            iteration += 1

        return (self._rows - self.total_error) / self._rows # Returns convergance
            
    def Test(self, data):
        ''' Takes Test data set to run against trained model
        '''
        hit = 0.0
        miss = 0.0
        for i in range(1, len(data) ):  # Skips Header row by starting at 1
            predicted = self.CalculateOutput(data[i])
            if ( predicted == data[i][ len(data[0])-1] ):
                hit += 1.0
            else :
                miss += 1.0
                
        #print ("Hits: ", hit, "   Misses: ", miss)
        return ( hit/(hit + miss) )
        
        
def Read_CSV(fileName):
    ''' Rawdata[0] = IV labels
        Rawdata[1:rowCt] = data values
        Rawdata[::][end] = DV classification label
    '''
    head = 0
    Rawdata = [[]] # 2 dimensional list of each data record
    rowCt = 0
    with open(fileName, newline='') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            a_list = []
            for i in row:
                a_list.append(i.replace("0", "-1")) # Replace 0's with -1's
            if (head == 1):
                Rawdata.append( list(map(float, a_list)) ) # avoids appending to empty array and instead fils in first cell
            else:
                Rawdata[0] = a_list
                head = 1
            rowCt += 1
    return Rawdata
    
def sigmoid(x):
    return 1.0/ (1.0+math.exp(-x))
        


            
if __name__ == "__main__":
    
    #if (len(sys.argv) != 4):
    #    raise Exception("Incorrect number of arguments. correct usage: ./perceptron <train> <test> <model>")
        
    TrainingData = "spambase-train.csv"
    #TrainingData = sys.argv[1]
    TestData = "spambase-test.csv"
    #TestData = sys.argv[2]
    Model = "model.txt"
    #Model = sys.argv[3]
    
    ## Train Perceptron
    
    # Read Training Data
    Rawdata = Read_CSV(TrainingData)
    print("Training Data --> Number of Rows: ", len(Rawdata), "    Number of Columns: ", len(Rawdata[0]), "\n")
    
    # Run Perceptron Trainer
    perc = Perceptron(Rawdata, 0)
    print("Training...")
    convergence = perc.Train()
    #print("convergence ratio = ", convergence)

    ## Test Perceptron
    TestData = Read_CSV(TestData)
    print("Test Data ------> Number of Rows: ", len(TestData), "    Number of Columns: ", len(TestData[0]), "\n")
    
    print("Testing...")
    accuracy = perc.Test(TestData)
    print("Model testing accuracy = ", accuracy )
    
    ## Write Out Weights  
    f = open(Model, "w")
    f.write( "Bias: " + str(perc._weights[-1]) )
    for i in range( len(perc._weights) -1 ):
            f.write("\n")
            f.write(perc._labels[i] + ": " + str(perc._weights[i]) )
    f.close()
