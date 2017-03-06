# Author: Joey M Nelson
# 2/3/2016
# Runtime Environment: Python 3.5

''' Machine Learning
        Binary Classification
            Perceptron Algorithm '''


# DO BIAS
# CALCULATE ACCURACY

import csv
import random
import math

class Perceptron(object):
    # Class Variables
    MAX_ITER = 1000
    LEARNING_RATE = 0.1
    TotalError = 1
    THETA = 0 # Used to predict classification if threshold of calculated output is > Theta


    def __init__(self, data, randomize_weights = 1):
        self._rows = len(data)
        self._cols = len(data[0])
        self._labels = data[0]
        
        self._data = data[1:self._rows] # Subtract header row from data
        
        # Create random weights if flag is set
        self._weights = [0]*self._cols
        if ( randomize_weights == 1 ):
            for i in range(self._cols-1):
                self._weights[i] = random.random() # 1 more for the bias this goes where the classifier would be
                
        self._weights[self._cols-1] = 0 # initialize Bias to 0
        
        self._TestData = [[]] # Create empty TestData data structure
        

        
    def CalculateOutput(self, inputs):
        """ This calculates the activation -1 or 1
            example: calculateOutput(inputs, weights):
            inputs = list of numerical/binary data; weights = list of weights for each data input
        """
        #print("length of inputs: ", len(inputs))
        if len(inputs) != len(self._weights): # Error checking
            raise ValueError('Input and weight list size mistmatch!: input len = ', len(inputs), "weight size = ", len(self._weights) )
            
        sum = 0.0
         
        #print ("len(inputs) ", len(inputs) )
        for i in range ( len(inputs) -1 ): # Minus 1 so we don't count the classification label
            sum += ( inputs[i] * self._weights[i] )
            
        sum += 1* self._weights[i+1] # add the bias
        
        return ( 1 if (sum >= self.THETA ) else -1 )
        
        
    def Train(self):
        ''' Trains model based on training data set
        '''
        total_error = 0
        outcalc = 0
        iteration = 0
        while (iteration < self.MAX_ITER):
            total_error = 0
            for i in range( len(self._data) ): # Loop through each data record
                
                outcalc = self.CalculateOutput(self._data[i])
                err = self._data[i][self._cols-1] - outcalc

                # Update Weights
                for u in range(self._cols):
                    self._weights[u] += self.LEARNING_RATE * err * self._data[i][u]
                self._weights[u] += self.LEARNING_RATE * err * 1 # Add bias
                
                total_error += math.fabs(err); ## convergence count
                
            if ( total_error == 0 ):
                print("CONVERGED!!!")
                break
            iteration += 1
        print("total_error = ", total_error)
            
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
                
        print ("Hits: ", hit, "   Misses: ", miss)
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
                #Rawdata[rowCt] = a_list
            if (head == 1):
                Rawdata.append( list(map(float, a_list)) ) # avoids appending to empty array and instead fils in first cell
            else:
                Rawdata[0] = a_list
                head = 1
            rowCt += 1
    return Rawdata

            



if __name__ == "__main__":
    
    
    ## Train Perceptron
    
    # Read Training Data
    Rawdata = Read_CSV("spambase\\spambase-train.csv")
    print("Rawdata -> Number of Rows: ", len(Rawdata), "\nNumber of Columns: ", len(Rawdata[0]), "\n")
    
    # Run Perceptron Trainer
    perc = Perceptron(Rawdata, 0)
    perc.Train()
    print("perc -> Number of Rows: ", perc._rows, "\nNumber of Columns: ", perc._cols, "\n")


    ## Test Perceptron
    TestData = Read_CSV("spambase\\spambase-test.csv")
    print("TestData -> Number of Rows: ", len(Rawdata), "\nNumber of Columns: ", len(TestData[0]), "\n")
    acc = perc.Test(TestData)
    
    
    print("Model Accuracy on Test Data = ", acc )
    #print(perc._weights)
    

    
   
    
   
    

    
