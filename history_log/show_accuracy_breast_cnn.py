import numpy as np
import glob
import matplotlib.pyplot as plt

result_list = []
result_list = glob.glob("*.p")
result_list.sort()
accuracy_list = []
loss_list = []


fig1 = plt.figure(1)
plt.figure(figsize=(20,10))
for i in  result_list:
    history = np.load(i, allow_pickle='TRUE')
    accuracy_list.append((history['accuracy']))
    loss_list.append((history['loss']))
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(accuracy_list)



fig2 = plt.figure(2)
plt.figure(figsize=(20,10))
for i in  result_list:
    history = np.load(i, allow_pickle='TRUE')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(loss_list)
plt.show()
