import numpy as np

class NaiveBayes():

    def __init__(self):
        self.X = None
        self.y = None
        self.classes = None
        self.parameter = None


    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(self.y)
        self.parameter = {}
        for i, c in enumerate(self.classes):
            X_c = np.array([ self.X[i] for i, yy in enumerate(self.y) if yy==c ])
            X_c_num = X_c.shape[0]
            X_c_mean = np.mean(X_c, axis=0)
            X_c_var = np.var(X_c, axis=0)
            self.parameter[c] = (X_c_num, X_c_mean, X_c_var)
        
    def predict(self, X):
        posterior = [ self.parameter[c][0] + self.pdfSum(X, c) for c in self.classes]

        return self.classes[np.argmax(posterior, axis=0)]

    def pdfSum(self, X, c):
        mu = self.parameter[c][1]
        sigma = self.parameter[c][2]
        nu = np.exp(-(X - mu) ** 2 / (2 * sigma))
        de = np.sqrt(2 * np.pi * sigma)
        return np.sum(np.log( nu/de ), axis=1)
        

if __name__=='__main__':
    import numpy as np
    x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
    y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])
    mm = NaiveBayes()
    mm.fit(x, y)
    print(mm.predict(x))

