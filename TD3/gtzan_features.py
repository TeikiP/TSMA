import librosa, librosa.display
import matplotlib.pyplot as plt
import pickle

# Dataset at 
# /net/cremi/pihanna/espaces/ens/AudioClassification/genres.tgz
pathAudio = '/media/hanna/TOSHIBA_EXT/Datasets/gtzan_genres/'

files=librosa.util.find_files(pathAudio,ext=['au'])

#pickle file
picklefile=open('gtzan_pickle.pickle', 'wb')

dataset=[]
for f in files:
    [signal, sr] = librosa.load(f, mono=True)
    waveform = signal[0]

    #Calcul d'un feature
    mfccs = librosa.feature.mfcc(signal, sr)
    # 20 mean
    m=mfccs.mean(axis=1)
    # 20 var
    v=mfccs.var(axis=1)
    
    x=f[47:].find('/')
    dataset.append([f[47+x+1:-9], [m, v]]) # a partir du 53eme char jusqu'a 9 depuis la fin
    print(f)

# pickle
pickle.dump(dataset,picklefile)
picklefile.close()
#librosa.display.waveplot(waveform)
#plt.show()

# verify / load
#picklefile=open('gtzan_pickle.pickle', 'rb')
#ds = pickle.load(picklefile)
#picklefile.close()
