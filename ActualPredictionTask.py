###################################################################
# Learning a linear regrssion model and using the learnt model to predict the affect of house being near the river on price
##################################################################

import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt

def predict(X, w, y = None):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample

    
    y_hat = np.dot(X,w)
    dif = np.subtract(y_hat,y)
    square = np.multiply(dif,dif)
    add = np.sum(square)
    
    loss  = add/(2*X.shape[0])

    Rdif = np.absolute(dif)
    Radd = np.sum(Rdif)
    
    risk  = Radd/(X.shape[0])
    
    return y_hat, loss, risk


def train(X_train, y_train, X_val, y_val):
    N_train = X_train.shape[0]
    N_val   = X_val.shape[0]

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val   = []

    w_best    = None
    risk_best = 10000
    epoch_best = 0
    
    for epoch in range(MaxIter):

        loss_this_epoch = 0
        for b in range( int(np.ceil(N_train/batch_size)) ):
            
            X_batch = X_train[b*batch_size : (b+1)*batch_size]
            y_batch = y_train[b*batch_size : (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            
            # Mini-batch gradient descent
            diff = np.subtract(y_hat_batch,y_batch)
            grad = np.dot(np.transpose(X_batch), diff)
            w_old = w
            w = np.subtract(w, np.multiply(alpha,grad))

        
        # monitor model behavior after each epoch
        # Compute the training loss by averaging loss_this_epoch
        losses_train.append(loss_this_epoch/batch_size)
        # Perform validation on the validation test by the risk
        _, _, val_perf = predict(X_val, w_old, y_val)
        risks_val.append(val_perf)
        # Keep track of the best validation epoch, risk, and the weights
        if val_perf<risk_best:
            risk_best = val_perf
            epoch_best = epoch
            w_best = w_old
    # Return some variables as needed
    return risk_best, epoch_best, w_best, risks_val, losses_train



############################
# Main code starts here
############################


X, y = datasets.load_boston(return_X_y=True)
y = y.reshape([-1, 1])
# X: sample x dimension
# y: sample x 1

############X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)



# Augment feature
X_ = X#np.concatenate( ( np.ones([X.shape[0],1]), X ), axis=1)
# X_: Nsample x (d+1)

# normalize features:
#mean_y = np.mean(y)
#std_y  = np.std(y)

#y = (y - np.mean(y)) / np.std(y)

#print(X.shape, y.shape) # It's always helpful to print the shape of a variable


# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_[:300]
y_train = y[:300]



X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
#mean_y = np.mean(y_train)
#std_y  = np.std(y_train)
y_train = (y_train - np.mean(y_train)) / np.std(y_train)
X_train = np.concatenate( ( np.ones([X_train.shape[0],1]), X_train ), axis=1)




X_val   = X_[300:400]
y_val   = y[300:400]
X_val = np.concatenate( ( np.ones([X_val.shape[0],1]), X_val ), axis=1)
X_test = X_[400:]
y_test = y[400:]
X_test = np.concatenate( ( np.ones([X_test.shape[0],1]), X_test ), axis=1)
#####################
# setting

alpha   = 0.001      # learning rate
batch_size   = 10    # batch size
MaxIter = 100        # Maximum iteration
decay = 0.0          # weight decay


############################################################
w = np.zeros([X_train.shape[1], 1])
y_hat, loss_batch, risk = predict(X_train,w,y_train)
##print(risk)
#print(loss_batch)


a,b,best_w ,risk_yaxis,losses_yaxis = train(X_train, y_train, X_val, y_val)
#print(a)
#print(b)
#print(best_w)
#print(d)
#print(e)
# Perform test by the weights yielding the best validation performance
_,_, test_perf = predict(X_test,best_w,y_test)
#print(test_perf)




# using learnt model to predict the affect of house being near the river on price


near_river_x = []
far_river_x = np.array([])
count = 0
for i in range(X_test.shape[0]):
     if (X_test[i,4]==0):
         #np.concatenate((far_river_x,X_test[i,:]),axis = 0)
         y_far,_,_ = predict(X_test[i,:],best_w,y_test[i,:])
         X_test[i,4] = 1
         y_near,_,_ = predict(X_test[i,:],best_w,y_test[i,:])

         count += y_near-y_far
         
     if (X_test[i,4]==1):
         #np.concatenate((far_river_x,X_test[i,:]),axis = 0)
         y_near,_,_ = predict(X_test[i,:],best_w,y_test[i,:])
         X_test[i,4] = 0
         y_far,_,_ = predict(X_test[i,:],best_w,y_test[i,:])

         count += y_near-y_far    
          




#this is the average predicted price difference if the house is near a river v/s not near a river 
print('Price difference for house near and not near a river = '+str(count/X_test.shape[0]))
#we can conclude that house near a river costs 

