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

#Unregularized regression

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


best_risk,best_epoch,best_w ,risk_yaxis,losses_yaxis = train(X_train, y_train, X_val, y_val)
print('best risk = '+str(best_risk))
print('best epoch = '+str(best_epoch))
#print(best_w)
#print(d)
#print(e)
# Perform test by the weights yielding the best validation performance
_,_, test_perf = predict(X_test,best_w,y_test)
print('Test perforamnce in best epoch = '+str(test_perf))

# Report numbers and draw plots as required.
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Validation Risk')
risk_epoch = range(1,101)
plt.plot(risk_epoch, risk_yaxis, color="blue")

#plt.legend()
plt.tight_layout()
plt.savefig('Risk_q2a.png')


plt.figure()
risk_epoch = range(1,101)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.plot(risk_epoch, losses_yaxis, color="blue")
#plt.legend()
plt.tight_layout()
plt.savefig('losses_q2a.png')




