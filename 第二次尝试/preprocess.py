import torch
import librosa
import numpy as np
import librosa.util as librosa_util
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn
from pathlib import Path
from matplotlib import pyplot as plt
from Config import Config


'''
preprocess.py
读取数据，进行预处理
1. STFT
2. 静音消除
3. 转化为梅尔谱
转换成(n_mel_channel=80, seq_len=1024)的npy文件存储到文件夹中，并生成data.txt，存储所有数据的路径
'''


## 下面这个类可以做 STFT特征提取
class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft
    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


# 对谱的动态压缩，取对数
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


# 对谱的反动态压缩
def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


#  求梅尔谱 算法
class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=11025):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        y = y.clamp(min=-1, max=1)
        # assert(torch.min(y.data) >= -1)
        # assert(torch.max(y.data) <= 1)
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output


def extract_mel_feature_bytaco(config: Config):
    '''
    :param hp:
    将数据集的语音 保持原始目录结构提取到 另一个文件夹
    :return:
    '''
    # 提取mel谱
    stftfunc = TacotronSTFT(filter_length=config.n_fft,  ###1024
                            hop_length=config.hop_length,  ## 256
                            win_length=config.win_length,  ###1024
                            n_mel_channels=config.n_mels,  ### 80
                            sampling_rate=config.sample_rate,  ###22050 different from tacotron2 22050
                            mel_fmin=config.f_min,  ## 0.0
                            mel_fmax=config.f_max)  ## 11025

    print("使用Taco2 mel 提取")

    src_wavp = Path(config.wav_datadir_name)
    for x in src_wavp.rglob('*'):
        if x.is_dir():
            Path(str(x.resolve()).replace(config.wav_datadir_name, config.feature_dir_name)).mkdir(parents=True, exist_ok=True)
    print("*" * 20)
    ########################################################################################
    wavpaths = [x for x in src_wavp.rglob('*.wav') if x.is_file()]
    ttsum = len(wavpaths)  # 总语音数量
    mel_frames = []
    k = 0
    for wp in wavpaths:
        k += 1
        the_wavpath = str(wp.resolve())
        the_melpath = str(wp.resolve()).replace(config.wav_datadir_name, config.feature_dir_name).replace('wav', 'npy')
        # wavform,_ = torchaudio.load(the_wavpath)
        wavform, _ = librosa.load(the_wavpath)
        wavform, _ = librosa.effects.trim(wavform, top_db=20)  ## 静音消除
        wavform = torch.FloatTensor(wavform).unsqueeze(0)  ## [1,length]
        mel = stftfunc.mel_spectrogram(wavform)
        mel = mel.squeeze().detach().cpu().numpy()  # [mel_dim, frames]
        np.save(the_melpath, mel)

        ## 统计mel谱的长度 信息
        mel_frames.append(mel.shape[-1])
        print("{}|{} -- mel_length:{}".format(k, ttsum, mel.shape[-1]))

    mean_len = sum(mel_frames) / len(mel_frames)
    max_fl = max(mel_frames)
    min_fl = min(mel_frames)
    print("*" * 100)
    print("Melspec length , Mean:{},Max:{},min:{}".format(mean_len, max_fl, min_fl))

    pass


# 作出 直方图。
def plot_hist_of_meldata(datadirname):
    datadirp = Path(datadirname)

    mellens = []
    wavpaths = [x for x in datadirp.rglob('*.npy') if x.is_file() ]

    for wavp in wavpaths:
        mel = np.load(str(wavp))
        mellens.append(mel.shape[-1])

    max_ = max(mellens)
    min_ = min(mellens)
    avg_ = int(sum(mellens)/len(mellens))
    print("max:{},min:{},avg:{}".format(max_,min_,avg_) )

    plt.figure()
    plt.title("MelLens_hist_" + "max:{},min:{},avg:{}".format(max_,min_,avg_))
    plt.hist(mellens)
    plt.xlabel("mel length")
    plt.ylabel("numbers")
    plt.savefig("Mel_lengths_hist" )
    plt.show()


if __name__=="__main__":
    # ################ preprocess  ###################################
    # # 将语音提取成meldata，用npy存储 （只存储 mel，无其他label）
    # config = Config() # 创建参数类对象
    # # preprocess_hp.set_preprocess_dir(wav_datadir_name,feature_dir_name) # 设置 源数据路径和目标路径
    # extract_mel_feature_bytaco(config) # 提取 mels特征


    ### 观察提取出来的melspec的时长分布图
    plot_hist_of_meldata("2018年数据集三合一_FSL")
    '''
    上面程序的结果为：
    ****************************************************************************************************
    Melspec length , Mean:199.09066666666666,Max:375,min:41
    '''