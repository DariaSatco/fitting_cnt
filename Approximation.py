import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from math import pi
import sys
import os


# # Читаем спектр образца из файла
#read data from file

il_datafile = sys.argv[1]
cnt_spectra = sys.argv[2:]

for datafile in cnt_spectra:

    string = datafile.split('/')
    name = string.pop()
    dirname = '/'.join(string)+'/'+'results'

    if not os.path.exists(dirname):
        os.makedirs(dirname)
# print(name,dirname)
# print(datafile)

    dataOCP=pd.read_csv(datafile, names=('wavelength', 'absorbance'), sep=';')
    dataOCP.drop(dataOCP.index[:2], inplace=True)
    #dataOCP.info() #control output
    dataOCP=dataOCP.astype(float) #convert object to float type
    #dataOCP.dtypes #show data types


# # Убираем ступеньки из спектра

    n1=dataOCP.loc[dataOCP['wavelength'] == 1800.0].index.values[0]
    n2=dataOCP.loc[dataOCP['wavelength'] == 860.0].index.values[0]

    dataOCP.loc[n2:,'step free']=dataOCP.loc[n2:,'absorbance']
    dataOCP.loc[n1:n2-1, 'step free']=dataOCP.loc[n1:n2-1,'absorbance']+(dataOCP.loc[n2,'absorbance']-\
         dataOCP.loc[n2-1,'absorbance'])
    dataOCP.loc[:n1-1,'step free']=dataOCP.loc[:n1-1,'absorbance']+(dataOCP.loc[n2,'absorbance']-\
         dataOCP.loc[n2-1,'absorbance']+dataOCP.loc[n1,'absorbance']-dataOCP.loc[n1-1,'absorbance'])

    #try to remove noise

    dataOCP['rol_mean'] = dataOCP['step free'].rolling(window=3,center=True).median()
    difference = np.abs(dataOCP['step free'] - dataOCP['rol_mean'])
    outlier_idx = difference > 0.01
    dataOCP.loc[outlier_idx, 'step free']=dataOCP.loc[outlier_idx, 'rol_mean']
#dataOCP.info()

# energy column
    h=4.135667e-15 #Planck constant
    cv=299792458 #speed of light
    nano=1e-9 #nano scaling
    dataOCP['energy']=h*cv/(dataOCP['wavelength']*nano)
#dataOCP.info()
#dataOCP


# # Спектр до вычета ионной жидкости

    plt.plot(dataOCP['energy'],dataOCP['step free'])
    plt.title('energy axis')
    plt.xlabel('energy, eV')
    plt.show()
    plt.plot(dataOCP['wavelength'],dataOCP['step free'])
    plt.title('wavelength axis')
    plt.xlabel('wavelength, nm')
    plt.show()


# # Читаем данные ионной жидкости

    dataIL=pd.read_csv(il_datafile, names=('wavelength','absorbance'),sep=';')
    dataIL.drop(dataIL.index[:2], inplace=True)
    #dataIL.info() #control output
    dataIL=dataIL.astype(float) #convert object to float type
    #dataIL.dtypes


    n1=dataOCP.loc[dataOCP['wavelength'] == 1800.0].index.values[0]
    n2=dataOCP.loc[dataOCP['wavelength'] == 860.0].index.values[0]

    dataIL.loc[n2:,'step free']=dataIL.loc[n2:,'absorbance']
    dataIL.loc[n1:n2-1, 'step free']=dataIL.loc[n1:n2-1,'absorbance']+dataIL.loc[n2,'absorbance']-\
         dataIL.loc[n2-1,'absorbance']
    dataIL.loc[:n1-1,'step free']=dataIL.loc[:n1-1,'absorbance']+dataIL.loc[n2,'absorbance']-\
         dataIL.loc[n2-1,'absorbance']+dataIL.loc[n1,'absorbance']-dataIL.loc[n1-1,'absorbance']
#dataIL.info()

    dataIL['energy']=h*cv/(dataIL['wavelength']*nano)
#dataIL.info()
#dataIL


# # Спектр ионной жидкости

    dataIL['rol_mean'] = dataIL['step free'].rolling(window=3,center=True).median()
    difference = np.abs(dataIL['step free'] - dataIL['rol_mean'])
    outlier_idx = difference > 0.01
    dataIL.loc[outlier_idx, 'step free']=dataIL.loc[outlier_idx, 'rol_mean']
#dataIL.info()

    plt.plot(dataIL['energy'],dataIL['step free'])
    plt.title('energy axis')
    plt.xlabel('energy, eV')
    plt.show()
    plt.plot(dataIL['wavelength'],dataIL['step free'])
    plt.title('wavelength axis')
    plt.xlabel('wavelength, nm')
    plt.show()


# # Вычитаем ионную жидкость из спектра образца


    dataOCP['data-IL']=dataOCP['step free']-dataIL['step free']

    plt.plot(dataOCP['energy'],dataOCP['data-IL'])
    plt.title('energy axis')
    plt.xlabel('energy, eV')
    plt.show()
    plt.plot(dataOCP['wavelength'],dataOCP['data-IL'])
    plt.title('wavelength axis')
    plt.xlabel('wavelength, nm')
    plt.show()


# # Ищем background

    # Lorentz-Fano model
    def LorentzFano(x, center,gammaL,numL, resonance,gammaF,q,numF):
        return numL/(pi*gammaL*(1.0+((x-center)/gammaL)**2)) +     (2*(x-resonance)/gammaF + q)**2/((2*(x-resonance)/gammaF)**2 + 1.0)*numF

    # Lorentz model
    def Lorentz(x, center,gammaL,numL):
        return numL/(pi*gammaL*(1.0+((x-center)/gammaL)**2))

    #Fano model
    def Fano(x, resonance,gammaF,q,numF):
        return numF*(2*(x-resonance)/gammaF + q)**2/((2*(x-resonance)/gammaF)**2 + 1.0)

    x=np.array(dataOCP['energy'])
    x1=np.concatenate((x[725:777], x[144:430]))
    y=np.array(dataOCP['data-IL'])
    y1=np.concatenate((y[725:777], y[144:430]))

    minparam=(4,0.1,0, 4,0.1,-10,0.005) #lower bound for approximation parameters
    maxparam=(6,3,10, 6,3,0,1)         #upper bound for approximation parameters

    LFopt, LFcov = curve_fit(LorentzFano, x1, y1, bounds=(minparam,maxparam), method='trf')
    Lopt, Lcov = curve_fit(Lorentz, x1, y1, bounds=(minparam[0:3],maxparam[0:3]), method='trf')
    Fopt, Fcov = curve_fit(Fano, x1, y1, bounds=(minparam[3:7],maxparam[3:7]), method='trf')

    print('Lorentz-Fano parameters')
    print(LFopt)

    print('Lorentz parameters')
    print(Lopt)

    print('Fano parameters')
    print(Fopt)

    plt.plot(x,y)

    LF=plt.plot(x, LorentzFano(x, *LFopt), color='red', label='Lorentz-Fano')

    #L=plt.plot(x, Lorentz(x, *Lopt), label='Lorentz')

    #F=plt.plot(x, Fano(x, *Fopt), color='green', label='Fano')

    plt.xlabel('energy, eV')
    plt.ylabel('absorbance')
    plt.legend()

    plt.show()

    u=np.array(dataOCP['wavelength'])

    plt.plot(u,y)

    LF=plt.plot(u, LorentzFano(x, *LFopt), color='red', label='Lorentz-Fano')

    #L=plt.plot(x, Lorentz(x, *Lopt), label='Lorentz')

    #F=plt.plot(x, Fano(x, *Fopt), color='green', label='Fano')

    plt.xlabel('wavelength, nm')
    plt.ylabel('absorbance')
    plt.legend()

    plt.show()


# # Записываем данные в таблицу

    dataOCP['Lorentz-Fano']=LorentzFano(x, *LFopt)
    dataOCP['Lorentz']=Lorentz(x, *Lopt)
    dataOCP['Fano']=Fano(x, *Fopt)

#calculate data - approximation
    dataOCP['data-LF']=dataOCP['data-IL']-dataOCP['Lorentz-Fano']
    dataOCP['data-L']=dataOCP['data-IL']-dataOCP['Lorentz']
    dataOCP['data-F']=dataOCP['data-IL']-dataOCP['Fano']
#dataOCP


    DLF=plt.plot(x, dataOCP['data-LF'], color='red', label='data - Lorentz-Fano')
    #DL=plt.plot(x, dataOCP['data-L'], label='Lorentz', color='blue')
    #DF=plt.plot(x, dataOCP['data-F'], label='Fano', color='green')
    plt.xlabel('energy, eV')
    plt.ylabel('absorbance')
    plt.legend()

    plt.show()

    DLF=plt.plot(u, dataOCP['data-LF'], color='red', label='data - Lorentz-Fano')
    #DL=plt.plot(x, dataOCP['data-L'], label='Lorentz', color='blue')
    #DF=plt.plot(x, dataOCP['data-F'], label='Fano', color='green')
    plt.xlabel('wavelength, eV')
    plt.ylabel('absorbance')
    plt.legend()

    plt.show()


    filename=dirname+'/'+name
    dataOCP.to_csv(filename, sep=';')
