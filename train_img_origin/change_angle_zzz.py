import numpy as np

data = np.loadtxt('../Dataset/label/label_304_close_2.csv')

# print(-86 // -90)

for i in range(len(data)):
    if np.abs(data[i][4] - data[i][5]) < 0.001:
        if data[i][6] > np.pi / 2:
            print('square change!')
            data[i][6] = data[i][6] - int(data[i][6] // (np.pi / 2)) * np.pi / 2
        elif data[i][6] < 0:
            print('square change!')
            data[i][6] = data[i][6] + (int(data[i][6] // (-np.pi / 2)) + 1) * np.pi / 2
    elif data[i][6] > np.pi:
        print('rectangle change!')
        # print(data[i])
        data[i][6] = data[i][6] - np.pi
    elif data[i][6] < 0:
        print('rectangle change!')
        # print(data[i])
        data[i][6] = data[i][6] + np.pi

for i in range(len(data)):
    if np.abs(data[i][4] - data[i][5]) < 0.001:
        if data[i][6] > np.pi / 2 or data[i][6] < 0:
            print('1 error')
    elif data[i][6] > np.pi or data[i][6] < 0:
        print('2 error')

cos = np.cos(2 * data[:, 6]).reshape(-1, 1)
sin = np.sin(2 * data[:, 6]).reshape(-1, 1)

data = np.concatenate((data, cos, sin), axis=1)

data = np.delete(data, [0, 1, 2, 3, 6], axis=1)
# for i in range(len(data)):
#     if data[i][0] < 0.00001:
#         print(data[i])
#         print('ssssssssssssssssss')

np.savetxt('../Dataset/label/label_304_close_2.csv', data)