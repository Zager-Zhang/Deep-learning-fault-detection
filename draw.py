from matplotlib import pyplot as plt
import numpy as np

# 修改成模型的名字
pic_name = 'Dae_2d_CWRUCWT'

dir_path = f'csv/{pic_name}'
step, acc_train = np.loadtxt(f'{dir_path}/acc_train.csv', unpack=True, delimiter=',', skiprows=1, usecols=(1, 2))
step, acc_val = np.loadtxt(f'{dir_path}/acc_val.csv', unpack=True, delimiter=',', skiprows=1, usecols=(1, 2))
step, loss_train = np.loadtxt(f'{dir_path}/loss_train.csv', unpack=True, delimiter=',', skiprows=1, usecols=(1, 2))
step, loss_val = np.loadtxt(f'{dir_path}/loss_val.csv', unpack=True, delimiter=',', skiprows=1, usecols=(1, 2))

fig, ax = plt.subplots()
plt.ylim(-0.1, 2.5)
ax.grid()
ax.plot(step, acc_train, label='train acc', color='red')
ax.plot(step, loss_train, label='train loss', color='green')
ax.plot(step, acc_val, label='val acc', color='blue')
ax.plot(step, loss_val, label='val loss', color='black')
ax.set_xlabel('epoch')
ax.set_ylabel('acc-loss')
ax.set_title(f'{pic_name}')
ax.legend()
plt.savefig(f'picture/{pic_name}')
plt.show()
