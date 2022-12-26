import pywt
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as scio
from scipy import signal

filename = '130'
datapath = f'cwru/12k Drive End Bearing Fault Data/{filename}.mat'

def show_STFT(data):
    fs = 12000  # sampling frequency
    amp = 2 * np.sqrt(2)
    per_seg_length = 200  # window length
    f, t, Zxx = signal.stft(data, fs, nperseg=per_seg_length, noverlap=0, nfft=per_seg_length, padded=False)
    plt.figure()
    ax2 = plt.subplot()
    ax2.pcolormesh(t, f, np.abs(Zxx))
    plt.title('STFT')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time [sec]')
    plt.savefig(f'picture_transform/STFT/STFT{filename}')
    plt.show()


def show_cwt(data):
    t = np.arange(0, 2048 / 12000, 1.0 / 12000)
    wavename = 'morl'
    totalscal = 256  # scale
    fc = pywt.central_frequency(wavename)  # central frequency
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(1, totalscal + 1)
    [cwtmatr_l, frequencies_l] = pywt.cwt(data, scales, wavename, 1.0 / 12000)  # continuous wavelet transform

    plt.figure()
    plt.contourf(t, frequencies_l, abs(cwtmatr_l), levels=np.linspace(0, 1.2, 40), extend='both')
    plt.title('CWT')
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.axis('on')
    # plt.savefig('test%.f.png' % (i), bbox_inches='tight', pad_inches=-0.1)  # 保存图像不显示白色边框
    plt.savefig(f'picture_transform/CWT/CWT{filename}')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    data = scio.loadmat(datapath)
    data = data[f'X{filename}_DE_time']
    print(data.shape)
    L = 2048
    data1 = np.zeros(L)
    for i in range(L):
        data1[i] = data[i]
    show_STFT(data1)
    show_cwt(data1)

