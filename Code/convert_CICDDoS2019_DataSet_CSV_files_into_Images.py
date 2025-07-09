import os
import pandas as pd
import numpy as np
import cv2
from sklearn import preprocessing

#DataPath of your CICDDOS CSV files.
path = 'G:\\TUI\\MasterThesisMain\\Datasets\\CICDDOS2019\\Secondday\\CSV-01-12\\syntraffic\\'
FilesList = os.listdir(path)

count = 0
dstpath = 'G:\\CICDDoS19_Scaled_1\\'
dstpath2 = dstpath + 'Benign\\'

for FileName in FilesList:
    print(FileName + ' DataFrame')

    dstpath3 = dstpath + FileName + 'Images\\'
    os.mkdir(dstpath3)
    #Reading the CSV into the DataFrame.
    df = pd.read_csv(DataPath + FileName)
    #Dropping unused columns.
    df.drop(labels=[
        'Unnamed: 0', 'Flow ID', ' Source IP', ' Source Port',
        ' Destination IP', ' Destination Port', ' Protocol', ' Timestamp',
        'SimillarHTTP', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags',
        'FIN Flag Count', ' PSH Flag Count', ' ECE Flag Count',
        'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate',
        ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
        ' RST Flag Count', ' Fwd Header Length.1', 'Subflow Fwd Packets',
        ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes'
    ],
            axis=1,
            inplace=True)
    #Replacing the infinity values with NaN.
    df = df.replace([np.inf, -np.inf], np.nan)
    #Dropping NaN values.
    df.dropna(inplace=True)

    print('Split Benign & Malicious Traffic')
    benign = df[df.loc[:, ' Label'] == 'BENIGN']
    Malicious = df[df.loc[:, ' Label'] != 'BENIGN']

    print('Normalize the Benign Traffic')
    benign.drop([' Label'], axis=1, inplace=True)

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 255))
    benign1 = minmax_scale.fit_transform(benign)
    benign = pd.DataFrame(benign1)

    print('Normalize the Malicious Traffic')
    Malicious.drop([' Label'], axis=1, inplace=True)

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 255))
    Malicious1 = minmax_scale.fit_transform(Malicious)
    Malicious = pd.DataFrame(Malicious1)

    #Check the size of Benign & Malicious Traffic
    r1 = int(len(benign) / 180)
    r2 = int(len(Malicious) / 180)

    print('Generate Benign Images')
    for i in range(0, r1):
        p = i * 180
        q = p + 60
        img = np.ones([60, 60, 3])
        img[:, :, 0] = benign.iloc[p:q, 0:60].values
        img[:, :, 1] = benign.iloc[q:q + 60, 0:60].values
        img[:, :, 2] = benign.iloc[q + 60:q + 120, 0:60].values

        ImageName = dstpath2 + FileName + str(i) + '_benign.png'
        cv2.imwrite(ImageName, img)
        print(i)

    print('Generate Malicious Images')
    for i in range(0, r2):
        p = i * 180
        q = p + 60
        img = np.zeros([60, 60, 3])
        img[:, :, 0] = Malicious.iloc[p:q, 0:60].values
        img[:, :, 1] = Malicious.iloc[q:q + 60, 0:60].values
        img[:, :, 2] = Malicious.iloc[q + 60:q + 120, 0:60].values

        ImageName = dstpath3 + FileName + str(i) + '_Malicious.png'
        cv2.imwrite(ImageName, img)
        print(i)
    print('Finished')
