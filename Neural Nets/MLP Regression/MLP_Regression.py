from scipy import optimize # learn more: https://python.org/pypi/scipy
import numpy as np # learn more: https://python.org/pypi/
from sklearn.metrics import r2_score
class neural_network(object):
  def __init__(self):
    # Define Hyper parameters
    self.InputLayerSize = 4
    self.OutputLayerSize = 1
    self.HiddenLayerSize = 8
    
    # Weights (parameters)
    self.W1 = np.random.randn(self.InputLayerSize, self.HiddenLayerSize)
    self.W2 = np.random.randn(self.HiddenLayerSize, self.OutputLayerSize)
    
  def forward(self, X):
    # propogate through neural_network
    self.z2 = np.dot(X, self.W1)
    self.a2 = self.sigmoid(self.z2)
    self.z3 = np.dot(self.a2, self.W2)
    yHat    = self.sigmoid(self.z3)
    return yHat
  
  def sigmoid(self, z):
    #Apply sigmoid activation function to scalar, vector, or matrix
    return 1/(1+np.exp(-z))
  
  def sigmoidprime(self, z):
    #Gradient of sigmoid
    return np.exp(-z)/((1+np.exp(-z))**2)    
  def costFunction(self, X, y):
    # compute cost for X and y using weights already stored in class.
    self.yHat = self.forward(X)
    J = 0.5*sum((y-self.yHat)**2)
    return J 
  def costFunctionPrime(self, X, y):
    # compute djdw1 and djdw2
    self.yHat = self.forward(X)
    delta3 =  np.multiply(-(y - self.yHat), self.sigmoidprime(self.z3))
    dJdW2 = np.dot(self.a2.T, delta3)
    delta2 = np.dot(delta3, self.W2.T) * self.sigmoidprime(self.z2)
    dJdW1 = np.dot(X.T,delta2)
    return dJdW1 , dJdW2 

# setter and getter methods    
  def getParams(self):
    # get W1 and W2 as a single vector
      params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
      return params
    
  def setParams(self, params):
    # set W1 and W2 from a single continuous vector
    W1_start = 0
    W1_end = self.HiddenLayerSize * self.InputLayerSize
    self.W1 = np.reshape(params[W1_start:W1_end], (self.InputLayerSize , self.HiddenLayerSize))
    W2_end = W1_end + self.HiddenLayerSize*self.OutputLayerSize
    self.W2 = np.reshape(params[W1_end:W2_end], (self.HiddenLayerSize, self.OutputLayerSize))
    
  def computeGradients(self, X, y):
    # Calculate gradients
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))  


class trainer(object):
  def __init__(self, N):
    self.N = N 
  
  def callBackF(self, params):
    self.N.setParams(params)
    self.J.append(self.N.costFunction(self.X, self.y))
  
  def costFunctionWrapper(self, params, X, y):
    self.N.setParams(params)
    cost = self.N.costFunction(X, y)
    grad = self.N.computeGradients(X,y)
    return cost, grad

  def train(self, X, y):
    #Make an internal variable for the callback function:
    self.X = X
    self.y = y 
    #Make empty list to store costs:
    self.J = [] 
    # initial params (weights)
    params0 = self.N.getParams()
    
    options = {'maxiter': 2000, 'disp' : True}
    
    _res = optimize.minimize(self.costFunctionWrapper, params0, jac = True, method = 'BFGS', \
                              args = (X, y), options = options, callback = self.callBackF)
    self.N.setParams(_res.x)
    self.optimizationResults = _res
    
    
# Main execution area   

#X = get np array of features
#Y = get np array of labels of continuous values
X = X/np.amax(X, axis = 0)
y = y/np.amax(y, axis = 0)
ANN = neural_network()
#yHat = ANN.forward(X)
#print(r2_score(yHat,y))
T = trainer(ANN)
T.train(X, y)
yHat = ANN.forward(X)
print(r2_score(yHat,y))
