import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.svm import SVC
import matplotlib.patches as mpatches

data = pd.read_csv('../data/training.csv')

svm_train = data[['recent_ave_rank', 'jockey_ave_rank']]
svm_train = np.array(pd.DataFrame(svm_train, dtype=np.float))
result = data.finishing_position
cnt = Counter(data.race_id)
svm_train_label = [int(result[i] <= np.floor(cnt[data.race_id[i]]/2)) for i in range(len(result))]

svc = SVC(kernel='linear')
svc.fit(svm_train, svm_train_label)

color_label = [(l==1)*'red'+(l==0)*'blue' for l in svm_train_label]
plt.scatter(svm_train[:, 0], svm_train[:, 1], c=color_label, zorder=10, edgecolor='k')
w = svc.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(0,15)
yy = a*xx - (svc.intercept_[0])/w[1]

margin = 1/np.sqrt(np.sum(svc.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a**2) * margin
yy_up = yy + np.sqrt(1 + a**2) * margin

plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

x_min = 1
x_max = 14
y_min = 0
y_max = 14

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = svc.predict(np.c_[XX.ravel(), YY.ravel()])

Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel('recent horse average rank')
plt.ylabel('recent jockey average rank')
plt.title('Vitualize SVM')
patch1 = mpatches.Patch(color='red' , label='upper 50%')
patch2 = mpatches.Patch(color='blue' , label='lower 50%')
plt.legend(handles=[patch1, patch2], loc='best')
plt.show()
