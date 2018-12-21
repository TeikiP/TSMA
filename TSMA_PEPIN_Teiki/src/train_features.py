import librosa, librosa.display
from os.path import expanduser
import os
import pickle

pathAudio = expanduser("~") + '/espaces/travail/audio/train'

files = librosa.util.find_files(pathAudio, ext=['mp3'])
filesAmount = str(len(files))

pickleFile = open('../data/pickle_train_new.pickle', 'wb')
corrFile = open('../data/corrupted_files_train_new.txt', 'w')

minFileSize = 50000;
i = 0
dataset=[]

for f in files:
  i += 1
  directoryIndex = f.rfind('/') + 1
  
  print(str(i) + "/" + filesAmount + " - " + f[directoryIndex:-4])
  
  if (os.stat(f).st_size > minFileSize):
    [signal, sr] = librosa.load(f, mono=True)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_var = mfcc.var(axis=1)
    
    rmse = librosa.feature.rmse(y=signal)
    rmse_mean = rmse.mean(axis=1)
    rmse_var = rmse.var(axis=1)
    
    tonn = librosa.feature.tonnetz(y=signal, sr=sr)
    tonn_mean = tonn.mean(axis=1)
    tonn_var = tonn.var(axis=1)
    
    zero = librosa.feature.zero_crossing_rate(y=signal)
    zero_mean = zero.mean(axis=1)
    zero_var = zero.var(axis=1)
    
    spbw = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    spbw_mean = spbw.mean(axis=1)
    spbw_var = spbw.var(axis=1)
    
    spct = librosa.feature.spectral_centroid(y=signal, sr=sr)
    spct_mean = spct.mean(axis=1)
    spct_var = spct.var(axis=1)
    
    spca = librosa.feature.spectral_contrast(y=signal, sr=sr)
    spca_mean = spca.mean(axis=1)
    spca_var = spca.var(axis=1)
    
    spfl = librosa.feature.spectral_flatness(y=signal)
    spfl_mean = spfl.mean(axis=1)
    spfl_var = spfl.var(axis=1)
    
    spro = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    spro_mean = spro.mean(axis=1)
    spro_var = spro.var(axis=1)
    
    stft = librosa.feature.chroma_stft(y=signal, sr=sr)
    stft_mean = stft.mean(axis=1)
    stft_var = stft.var(axis=1)
    
    ccqt = librosa.feature.chroma_cqt(y=signal, sr=sr)
    ccqt_mean = ccqt.mean(axis=1)
    ccqt_var = ccqt.var(axis=1)
    
    cens = librosa.feature.chroma_cens(y=signal, sr=sr)
    cens_mean = cens.mean(axis=1)
    cens_var = cens.var(axis=1)
    
    mels = librosa.feature.melspectrogram(y=signal, sr=sr)
    mels_mean = mels.mean(axis=1)
    mels_var = mels.var(axis=1)
    
    dataset.append([f[directoryIndex:-4], [mfcc_mean, mfcc_var], [rmse_mean, rmse_var], [tonn_mean, tonn_var], [zero_mean, zero_var], [spbw_mean, spbw_var], [spct_mean, spct_var], [spca_mean, spca_var], [spfl_mean, spfl_var], [spro_mean, spro_var], [stft_mean, stft_var], [ccqt_mean, ccqt_var], [cens_mean, cens_var], [mels_mean, mels_var]])
    
  else:
    corrFile.write(f[directoryIndex:-4] + "\n")

pickle.dump(dataset,pickleFile)
pickleFile.close()
corrFile.close()

print('end')
