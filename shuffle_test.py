import numpy as np

# 随机数置乱

for time in range(3):
    idx = [i for i in range(256*256)]
    np.random.seed(0)
    np.random.shuffle(idx)
    np.reshape(idx,[256,256])

    print(idx[:100])

values = [i for i in range(256*256)]
new_values = np.zeros([256*256,1])
idx = [i for i in range(256*256)]
np.random.seed(0)
np.random.shuffle(idx)
for i in range(256*256):
    new_values[idx[i]] = values[i]
revert = np.zeros([256*256,1])
for i in range(256*256):
    revert[i] = new_values[idx[i]]

print()
print(revert[:100])
