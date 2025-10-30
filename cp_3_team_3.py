
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Applies sigmoid function to turn a number into a probability between 0 and 1
def calculate_sigmoid(z):
    #Clip to make sure that we dont have like e^1000 which is equal to infinity
    """How do I pick where to clip???"""
    z=np.clip(z,-1000,1000)
    #Epsilon to make sure we're not dividing by zero
    epsilon= 1e-15
    #Sigmoid equation, converts a number to a probability between 0 and 1
    sigmoid=1/(1+np.exp(-z)+epsilon)
    #Makes sure that sigmoid actually falls in between 0 and 1 and not exactly 0 or 1
    sigmoid=np.clip(sigmoid,epsilon,1-epsilon)
    return sigmoid

#Find the predicted probabilities for the features
def predict_proba(X,w,b):
    #Changes X to an array if not already one
    X=np.array(X)
    #Makes sure matrix X has dimensions (m,n)
    if X.ndim==1:
        #If the array is one dimensional, then reshape it to be two dimensional so it has the proper dimensions for matrix multiplication
        X=X.reshape(1,-1)
    else: 
        #Otherwise, if the array is already multi-dimensional, then you don't have to change anything
        X=X
    #Makes sure that w is in the shape (n,1) also so that its the correct dimensions for matrix multiplication
    w=w.reshape(-1,1)
    #Find z for sigmoid-->>
        #Basically taskes each feature and applie a weight to it and tehn adds the bias to it to find the linear combination of inputs and weigts
    z=X@w+b
    #Find the porbablity of z from 0-1
    proba=calculate_sigmoid(z)
    return proba

#Finds the model's predictions for each image
def predict_labels(X,w,b,threshold=0.5):
    #Gets probabilities using predict_proba for matrix X
    proba=predict_proba(X,w,b)
    #Compares probability and threshold, if true, label =1, if false, label=0-->>
        #Threshold is just the cutoff for if the model should predict 1 or 0
    labels=(proba>=threshold).astype(int)
    return labels

#Finds the loss(how wrong the model's predictions are) and gradients(slope of loss with respect to w and b)
def compute_loss_and_grads(X,y,w,b):
    #Mkaes sure that log(0) never happens
    epsilon=1e-15
    #Number of training examples
    m=X.shape[0]
    #Makes y a row vector instread of just a number to (m,1) so that it matches the dimensions of y_pred
    y=y.reshape(-1,1)
    #Find predicted probabilities using predict_proba
    y_pred=predict_proba(X,w,b)
    #Clip predicted values so that we don't have to deal with log(0) which is equal to -infinity
    y_pred_clipped=np.clip(y_pred,epsilon,1-epsilon)
    #Find binary cross entropy loss-->>
        # it basically measuers how well our model is predicting 
        # whether a sign is a stop sign or a right turn arrow sign. 
        # This model also hevaily penalizes confident wrong predictions
    loss=(-(1/m))*(np.sum((y*np.log(y_pred_clipped))+(1-y)*(np.log(1-y_pred_clipped))))
    #Find weight gradient-->>
        #Weights basically control how important a feature is
        #X.T is just matrix X transposed, so since X=(m,n), X.T=(n,m)--??
            #Needed becasue otherwise matrix multiplication wont work, (m,n)x(m,1) doesn't work so you have to do (n,m)x(m,1)-->> inner dimmensions have to match
    dw=(1/m)*(X.T@(y_pred-y))
    #Find bias gradient-->>
        #Bias basically controls the overall tendency of the model to predict a 1 or 0 regardless of featrues
    """why is b not just 0.5?"""
    db=(1/m)*(np.sum(y_pred-y))
    return loss,dw,db

#Train model, adjust the weights and bias over a bunch of interations so we can minimize loss
def train_logistic_regression(X_train,y_train,X_val=None,y_val=None,learning_rate=0.01,num_iterations=1000):
    #Number of features/columns in array X_train
    n_features=X_train.shape[1]
    #Initialize w and b 
    w=np.random.randn(n_features,1)*0.01
    b=0.0
    #Create empty lists for train and validation losses that we use to plot later
    train_losses=[]
    val_losses=[]
    #Loop to update weights and bias for the given number of iterations
    for i in range(num_iterations):
        #Finds the current loss and gradients
        loss,dw,db=compute_loss_and_grads(X_train,y_train,w,b)
        #Update the weight
        w=w-(learning_rate*(dw))
        #Update the bias
        b=b-(learning_rate*(db))
        #Add training loss history to the training loss list
        train_losses.append(loss)

        #Checks if the user added validation data, if yes execute this code
        if X_val is not None and y_val is not None:
            #We only care about the loss data, so we only call the loss info from the outputs of the compute_loss_and_grads UDF
                #The _'s are just placeholders
            val_loss,_,_=compute_loss_and_grads(X_val,y_val,w,b)
            #Adds validation loss history to validation loss list
            val_losses.append(val_loss)
        #If the user doesn't input thinsg for X_val and y_val, then add "None" to the validation loss list so that the list lengths can be consistent even without validation data
        else:
            val_losses.append(None)


##############################################
        #add code to check validation performacne#
        "what does that even mean????"
############################################## 
    return w,b,train_losses,val_losses

#Measuers how accurate the model is
def calculate_metrics(predicted_labels,true_labels):
    #Finds the fraction of correct pridictions that the model makes, aka accuracy of the model-->>
        #Basically, predicted_labels==true_labels returns an array of true 
        #and false values which then is converted to 0s and 1s and then 
        #np.mean finds the mean of the array
    accuracy=np.mean(predicted_labels==true_labels)
    #Finds error rate which is just how often the model makes incorrect predictions
    error_rate=1-accuracy
    return accuracy, error_rate

#Finds preditctions and calculates metrics at the same time in one function for efficency and convinence 
def evaluate_logistic_regression(X,y,w,b):
    #Find the predicted labels from the model with predict_labels UDF
    predicted_labels=predict_labels(X,w,b)
    #Find accurcy and error rate of the model using calculate_metrics UDF
    accuracy,error_rate=calculate_metrics(predicted_labels,y)
    return accuracy,error_rate
  
#Load .csv data
#Split dataset into training, validation, and test sets (60/20/20)
#Standardize all the features so that all teh features are on a similar scale
#Train the model using the training set (model learns from the dataset, adjusts parameters to minimize loss, memorize patterns in the data)
#Evaluate the model on the validation and test sets (validation is when basically were checking how well the model generalizes, its no longer updating weights...testing is when were measuring the final performance of the model, gives unbiased estimate of real world accuracy)
#Plot the loss curves
def main():
    #Turn .csv file into a pandas data frame which is just a tbale that 
        #data can be extracted from to be used when training our model
    data=pd.read_csv("img_features.csv")
    #Drop the column that has path and class id info from the data frame, 
        #so keep all the features data
    X=data.drop(columns=["Path","ClassId"]).values
    #Extracts the class id column and info since that's the true label of each image
    y=data["ClassId"].values

    #m is equal to the number of samples in our dataset, 
        #its the number of rows in our dataset
    m=X.shape[0]
    #Shuffle dataset before splitting it into train data, validation data, and testing data
        #Basically, np.random.permutation(m) creates a shuffled array of integers from 0 to m-1. 
        #So this shuffles the order that we use the dataset, or aka shuffles the dataset
    indicies=np.random.permutation(m)
    #Training data will be 60% of the dataset
    train_end=int(0.6*m)
    #Validation data will be 20% of the data set and the remaining 20% will be testing data
        #It's 0.8*m because it ends at the 80% mark of the dataset but the first 60% is already 
        #training data so the remaining 20% will be the validation data, thats how we get the 60/20/20 split
    val_end=int(0.8*m)

    #Splits the dataset into train data, validation data, and test data
    #Train data goes from the begining of the dataset to the 60% aka train_end
    train_idx=indicies[:train_end]
    #Validation data goes from the 60% mark to the 80% make aka from train_end to val_end
    val_idx=indicies[train_end:val_end]
    #Test data goes from the 80% mark to the end of the dataset aka val_end to the end of the dataset
    test_idx=indicies[val_end:]

    #Splits dataset into training, validation and test sets using the indices we created earlier
    #For all the X..., y... variables, X is the features data and y is the values, like the 1s and 0s for stop signs and right turn arrows
    X_train,y_train=X[train_idx], y[train_idx]
    X_val,y_val=X[val_idx],y[val_idx]
    X_test,y_test=X[test_idx],y[test_idx]

    # Calculate statistics from training data
    #Mean value of each featyre acorss all training data
        #Axis=0 is basically saying that we are caluclating the mean column-wise or per column, 
        #like for each feature seperately since our features are split onto columns
    feature_means = np.mean(X_train, axis=0)
    #Standard deviarion of each feature across all training data
        #Axis=0 is saying the same as before, calculate column-wise
    feature_stds = np.std(X_train, axis=0)
    #Standardize training data
    #Standardize the training features by transforming each feature to have a mean of about 0 and a standard deviation of about 1 therefore standardizing it
    X_train_std = (X_train - feature_means) / feature_stds
    # Apply same transformation to validation/test data
    X_val_std = (X_val - feature_means) / feature_stds
    X_test_std = (X_test - feature_means) / feature_stds


    #Trian model
    #Returns the learned weights for each feature (w), the learned bias (b), and the list of loss values during training (train_losses)
    w,b,train_losses,val_losses=train_logistic_regression(X_train_std,y_train,X_val=X_val_std,y_val=y_val,learning_rate=0.01,num_iterations=5000)

    #Evaluate model
    #returns the accuracy and error of the model on the validataion and test sets
    val_accuracy,val_error=evaluate_logistic_regression(X_val_std,y_val,w,b)
    test_accuracy,test_error=evaluate_logistic_regression(X_test_std,y_test,w,b)

    #Print the accurcy and error for the validation and test sets
    print(f"\nValidation Accuracy: {val_accuracy:.4f}, Validation Error: {val_error:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}, Test Error: {test_error:.4f}")

    #Plot Training Loss Graph
    plt.plot(train_losses,label="Training Loss")
    plt.plot(val_losses,label="Validation Loss")
    plt.title("Training Progres")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()






  