#%%
import os
import pandas as pd
import numpy as np
import cv2
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from pandasgui import show

#path = 'G:\\TUI\\MasterThesisMain\\Datasets\\CICDDOS2019\\Firstday\\CSV-01-12\\Synsplit\\'
#path = 'G:\\TUI\\MasterThesisMain\\Datasets\\CICDDOS2019\\Secondday\\CSV-03-11\\UDPfull\\'
path = 'E:\\CICDDoS2019\\MS\\UDP\\CSV\\'

listOfFiles = os.listdir(path)

#dstpath = 'G:\\CICDDoS19_IMGs_1\\'
dstpath = 'E:\\CICDDoS2019\\MS\\UDP\\IMAGES\\'

#dstpath = 'G:\\CICDDoS19_IMGs_2\\'

for fname in listOfFiles:
    print(fname + ' dataframe')

    dstpath2 = dstpath + fname
    os.mkdir(dstpath2)
    # dstpath3 = dstpath + 'attacks\\'
    # os.mkdir(dstpath3)

    print('--- Reading File into DataFrame ---')
    df = pd.read_csv(path + fname)
    df.info()
    # show(df)
    print(df['Label'].unique())

    print('--- Dropping Useless Features')
    df.drop(labels=[
        'Active Max', 'Active Min', 'Active Mean', 'Idle Min', 'Idle Max',
        'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'URG Flag Cnt',
        'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'ECE Flag Cnt',
        'Pkt Len Min', 'Fwd IAT Tot', 'Bwd Byts/b Avg', 'Bwd Seg Size Avg',
        'Fwd Pkts/b Avg', 'Fwd Byts/b Avg', 'ACK Flag Cnt', 'Fwd Pkts/s',
        'Bwd Pkt Len Std', 'Subflow Bwd Byts', 'Subflow Bwd Pkts',
        'Fwd IAT Std', 'Bwd IAT Std', 'Bwd IAT Max', 'Fwd IAT Mean',
        'Active Std', 'Pkt Len Mean', 'Fwd IAT Max', 'Fwd Blk Rate Avg',
        'Fwd Seg Size Avg'
    ],
            axis=1,
            inplace=True)
    print('df.shape: ', df.shape)

    #Replace all values = Infinity with NAN
    #df = df.replace('Infinity', np.NaN)
    df = df.replace([np.inf, -np.inf], np.nan)

    # drop rows with missing values
    df.dropna(inplace=True)
    print('df.shape After dropping Useless Features: ', df.shape)

    # profile = ProfileReport(df, title="Pandas Profiling Report")
    # profile.to_file('your_report_name.html')

    #print('Separating Normal & Attack Traffic')
    #normal = df[df.loc[:, 'Label'] == 'BENIGN']
    normal = df[df.loc[:, 'Label'] != 'BENIGN']

    print('Normalizing the Normal Traffic')
    normal.drop(['Label'], axis=1, inplace=True)
    print('normal.shape After dropping Label column: ', normal.shape)

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 255))
    normal1 = minmax_scale.fit_transform(normal)
    normal = pd.DataFrame(normal1)

    # print('Normalizing the Attack Traffic')
    # attack.drop([' Label'], axis=1, inplace=True)
    # print('attack.shape After dropping Label column: ', attack.shape)

    # minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 255))
    # attack1 = minmax_scale.fit_transform(attack)
    # attack = pd.DataFrame(attack1)

    print('Checking size of Normal & Attack Traffic')
    r1 = int(len(normal) / 180)
    #r2 = int(len(attack) / 180)

    print('--- Generating Normal Images ----')
    for i in range(0, r1):
        p = i * 180
        q = p + 60
        img = np.zeros([60, 49, 3])

        img[:, :, 0] = normal.iloc[p:q, 0:49].values
        #blue_channel = img[:, :, 0]
        #cv2.imwrite('G:/cv2-blue-channel.png', blue_channel)  #write blue channel to greyscale image

        img[:, :, 1] = normal.iloc[q:q + 60, 0:49].values
        #green_channel = img[:, :, 1]
        #cv2.imwrite('G:/cv2-green-channel.png', green_channel) #write green channel to greyscale image

        img[:, :, 2] = normal.iloc[q + 60:q + 120, 0:49].values
        #red_channel = img[:, :, 2]
        #cv2.imwrite('G:/cv2-red-channel.png', red_channel) #write red channel to greyscale image

        imgName = dstpath2 + fname + str(i) + 'benign.png'
        cv2.imwrite(imgName, img)
        print(i)

    # print('--- Generating Attack Images ----')
    # for i in range(0, r2):
    #     p = i * 180
    #     q = p + 60
    #     img = np.zeros([60, 60, 3])
    #     img[:, :, 0] = attack.iloc[p:q, 0:60].values
    #     img[:, :, 1] = attack.iloc[q:q + 60, 0:60].values
    #     img[:, :, 2] = attack.iloc[q + 60:q + 120, 0:60].values

    #     imgName = dstpath3 + fname + str(i) + '_attack.png'
    #     cv2.imwrite(imgName, img)
    #     print(i)
    print('--- done ---')
print('--- All Done ---')
# %%
