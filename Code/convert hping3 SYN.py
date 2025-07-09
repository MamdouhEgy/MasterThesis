import os
import pandas as pd
import numpy as np
import cv2
from sklearn import preprocessing

path = 'D:\\VirtualBox\\Shared\\ilmenau\\ryu\\FlowAttackSyn\\'

listOfFiles = os.listdir(path)

dstpath = 'G:\\DDoS_Syn_IMGs\\'

for fname in listOfFiles:
    print(fname + ' dataframe')

    os.mkdir(dstpath)

    print('--- Reading File into DataFrame ---')
    df = pd.read_csv(path + fname)
    df.info()

    normal = df[df.loc[:, 'ip_proto'] == 6]
    print('normal.shape : ', normal.shape)

    normal.drop(['flow_id'], axis=1, inplace=True)
    normal.drop(['ip_src'], axis=1, inplace=True)
    normal.drop(['ip_dst'], axis=1, inplace=True)

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 255))
    normal1 = minmax_scale.fit_transform(normal)
    normal = pd.DataFrame(normal1)

    print('Checking size of Normal & Attack Traffic')
    r1 = int(len(normal) / 180)

    print('--- Generating Syn Attack Images ----')
    for i in range(0, r1):
        p = i * 180
        q = p + 60
        img = np.zeros([60, 18, 3])

        img[:, :, 0] = normal.iloc[p:q, 0:18].values
        img[:, :, 1] = normal.iloc[q:q + 60, 0:18].values
        img[:, :, 2] = normal.iloc[q + 60:q + 120, 0:18].values

        imgName = dstpath + fname + str(i) + 'Syn.png'
        cv2.imwrite(imgName, img)
        print(i)

    print('--- done ---')
print('--- All Done ---')