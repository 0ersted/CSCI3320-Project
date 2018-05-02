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
        de = np.sqrt(2 * np.pi * sigma)+ 1e-5
        return np.sum(np.log( nu/de ), axis=1)
        

if __name__=='__main__':
    from classification import Classification

    clf = Classification()
    mm = NaiveBayes()
    mm.fit(clf.X_train, clf.y_train)
    print(mm.predict(clf.X_test))

