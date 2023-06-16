#!/usr/bin/python3.8
# -*- coding: utf-8 -*-


from bayes_opt_WEI_test import BayesianOptimization
from bayes_opt_WEI_test import UtilityFunction
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.font_manager as fm
import subprocess
import re
import matplotlib.pyplot as plt
import time
import os
import os.path
import warnings
import copy
import threading
import decimal
import sys
import math
sys.path.append("/home/jihwan/simulation/powerOPTwithConst/powerPredict/Post/predict_python")
import changeLayout as CHL
import TRKBcell_onNumChange_pexnetlist as CHnTRK
import findandchange as CHW
import extractCoordinate as Ecoor
import getSAvos_Post_c1 as SAV
import linearPredict as lp
import RangePredict as rp
import removefile as rmf
import ParamGrep_ver2 as pg

import readytopex as rtp
    
now=datetime.now()
today=now.strftime('%Y-%m-%d_%H:%M:%S')
start=time.time()
#####VSAT & PTWG fitting
np.random.seed = 2
bo_iter = 300
bo_pre_iter = 300
GPlearn = False #gplearn은 multicorner수정 안돼있음
initlearn = True

autoParam= False
LRmodel=False
if (GPlearn == False and LRmodel==True):
    sys.exit("When GPlearn is False, LRmodel cannot be used.")
#몇번에 한번 layout 수정 후 pex.netlist 추출
pex_netlist_count = 20000


# ['PCH1.MPC<0>', 'WD1.IWD_L.M0', 'WD1.IWDD_L.M2', 'WD1.IWD_L.M2', 'WD1.IWDD_L.M0', 'SA1.I1.M0', 'SA1.I1.M2', 'RM1.I16.M0','RM1.I16.M2'
#                   ,'SA1.MPD_L','SA1.MPU_L','SA1.MSAPCH_L','SA1.MPG_L','CNTL.I57.M2' ]

# ['PCH1.MPC<0>', 'WD1.IWD_L.M0', 'WD1.IWDD_L.M2', 'WD1.IWD_L.M2', 'WD1.IWDD_L.M0', 'SA1.I1.M0', 'SA1.I1.M2', 'RM1.I16.M0','RM1.I16.M2'
#                  ,'SA1.MPD_L','SA1.MPU_L','SA1.MSAPCH_L','CNTL.I57.M2' ]

path="/home/jihwan/simulation/powerOPTwithConst/powerPredict/Post/MultiCorner_nominal"
os.chdir(path)#path
instSize = '512X64'
rows = int(int(instSize.split('X')[0])/4)
columns = int(instSize.split('X')[1])*4

[row_base1, col_base1]=[rows,32]#128X8
[row_base2, col_base2]=[rows,128]#128X32
[row_target, col_target]=[rows,columns]#128X128

instanceSize1='{}X{}'.format(int(row_base1*4),int(col_base1/4))
instanceSize2='{}X{}'.format(int(row_base2*4),int(col_base2/4))
targetInstanceSize='{}X{}'.format(int(row_target*4),int(col_target/4))

CHnTRK.TRKCH(instanceSize1, 20)

# if(GPlearn==False):
#     rtp.ReadytoPex(instanceSize1, instanceSize2,instSize)

if(autoParam):
    userParamList=pg.grepPar(instanceSize1,24,instSize)
    userParamList.remove('PCH1.M28')
    #userParamList.remove('SA1.MSAOUT_L__2')
    #userParamList.remove('SA1.MSAOUT_R__2')
    #userParamList.append('SA1.MPU_L')
    print(userParamList)
    # width=CHW.widthFI2(instSize,userParamList)
    # width2=CHW.widthFI2(instSize,userParamList_cut)
    # kk=lawidth=CHW.widthFI2(instSize,last)
#%%
else:
    userParamList = ['PCH1.MPC<0>', 'WD1.IWD_L.M0', 'WD1.IWDD_L.M2', 'WD1.IWD_L.M2', 'WD1.IWDD_L.M0', 'SA1.I1.M0', 'SA1.I1.M2', 'RM1.I16.M0','RM1.I16.M2'
                 ,'SA1.MSAOUT_L','SA1.MPU_L','SA1.MSAPCH_L','SA1.MPG_L', 'SA1.MFOOT']
#%%
tcycleConst = 5.65305e-10
WMConst= 9e-1   #90%
TaccessConst= 3.7687e-10
RMConst = 0


rp.rangePredict('1024X80',False)

ref = instanceSize1+"_ref.sp"
ref_Vth0 = instanceSize1+"_Vth0_ref.sp"
runsp = instanceSize1+"_optimizing.sp"
runsp_Vth0 = instanceSize1+"_Vth0_optimizing.sp"

ref2 = instanceSize2+"_ref.sp"
ref2_Vth0 = instanceSize2+"_Vth0_ref.sp"
runsp2 = instanceSize2+"_optimizing.sp"
runsp2_Vth0 = instanceSize2+"_Vth0_optimizing.sp"

lisFile = runsp.split(sep='.')[0]
lisFile_Vth0 = runsp_Vth0.split(sep='.')[0]
lisFile2 = runsp2.split(sep='.')[0]
lisFile2_Vth0 = runsp2_Vth0.split(sep='.')[0]

leakage1 = instanceSize1+"_leakage.sp"
leakage2 = instanceSize2+"_leakage.sp"
leakage_lis1 = leakage1.replace('.sp','.log')
leakage_lis2 = leakage2.replace('.sp','.log')

ref_c1 = instanceSize1+"_ref_c1.sp"
ref_Vth0_c1 = instanceSize1+"_Vth0_ref_c1.sp"
runsp_c1 = instanceSize1+"_optimizing_c1.sp"
runsp_Vth0_c1 = instanceSize1+"_Vth0_optimizing_c1.sp"

ref2_c1 = instanceSize2+"_ref_c1.sp"
ref2_Vth0_c1 = instanceSize2+"_Vth0_ref_c1.sp"
runsp2_c1 = instanceSize2+"_optimizing_c1.sp"
runsp2_Vth0_c1 = instanceSize2+"_Vth0_optimizing_c1.sp"

spnameList = [runsp, runsp_Vth0, runsp2, runsp2_Vth0,runsp_c1, runsp_Vth0_c1, runsp2_c1, runsp2_Vth0_c1] #sp file name list for parallel simulation
lisFile_c1 = runsp_c1.split(sep='.')[0]
lisFile_Vth0_c1 = runsp_Vth0_c1.split(sep='.')[0]
lisFile2_c1 = runsp2_c1.split(sep='.')[0]
lisFile2_Vth0_c1 = runsp2_Vth0_c1.split(sep='.')[0]

leakage1_c1 = instanceSize1+"_leakage_c1.sp"
leakage2_c1 = instanceSize2+"_leakage_c1.sp"
leakage_lis1_c1 = leakage1_c1.replace('.sp','.log')
leakage_lis2_c1 = leakage2_c1.replace('.sp','.log')
leakageList = [leakage1, leakage2,leakage1_c1, leakage2_c1]

minWidth = 0.1#u # 100e-9

#%%


if ((os.path.isfile('ParamName_data.txt')) and GPlearn==True):
    with open('ParamName_data.txt',"r") as pamr:
        paramNameList=pamr.readlines()
        for pmr in range(0,len(paramNameList)):
            paramNameList[pmr]=paramNameList[pmr].strip()
    paramNameList.remove('Ntrk')
else:
    paramNameList=userParamList
    
add_pa=list(set(userParamList)-set(paramNameList))
del_pa=list(set(paramNameList)-set(userParamList))
for ad in range(0,len(add_pa)):
    paramNameList.append(add_pa[ad])

fingnumdict1=Ecoor.finNumExtract(instanceSize1,paramNameList)
fingnumdict2=Ecoor.finNumExtract(instanceSize2,paramNameList)
fingnumlist1 = []
fingnumlist2 = []
for i in paramNameList:
    fingnumlist1.append(fingnumdict1[i])
    fingnumlist2.append(fingnumdict2[i])
initFinDict = copy.deepcopy(fingnumdict1)
initFinList = list(initFinDict.values())


if(rows > 128):
    initTRKnum = 11
elif(rows > 15):
    initTRKnum = 16
else:
    initTRKnum = 8


old_width_dict1=CHW.widthFI2(instanceSize1,userParamList)
old_width_dict2=CHW.widthFI2(instanceSize1,paramNameList)
add_dict=list(set(old_width_dict1)-set(old_width_dict2))
#%%
for ad in range(0,len(add_dict)):
    old_width_dict2.update({add_dict[ad]:old_width_dict1[add_dict[ad]]})
old_width_dict=copy.deepcopy(old_width_dict2)

initValue={}
bnd={}
for par in range(0,len(paramNameList)):
    initValue.update({paramNameList[par]:old_width_dict[paramNameList[par]]*initFinList[par]})
    bnd.update({paramNameList[par]:(minWidth, initValue[paramNameList[par]])})
initValue.update({"Ntrk":initTRKnum})
bnd.update({"Ntrk":(initTRKnum, rows)})


if(len(del_pa)!=0):
    for dp in range(0,len(del_pa)):
        bnd[del_pa[dp]]=(initValue[del_pa[dp]],initValue[del_pa[dp]])


paramList=[]

for i in initValue:
    paramList.append(i)
    
paramNum = len(paramList)

powerBO = BayesianOptimization(f=None, pbounds=bnd, random_state=27)
TcycleGPR = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=27,
        )
WMGPR = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=27,
        )

TaccessGPR = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=27,
        )
RMGPR = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=27,
        )
PowerLR=LinearRegression()
TcycleLR=LinearRegression()
WMLR=LinearRegression()
TaccessLR=LinearRegression()
RMLR=LinearRegression()


VosMean = re.compile('VosCH1')
VosSigma = re.compile('VosCH2')

def runFineSim(spFileName):
    os.system("runFinesim {}".format(spFileName))

def measLeakage(widthdict) : 
    nTRK_leak=round(widthdict['Ntrk'])
    CHnTRK.TRKCH(instanceSize1, nTRK_leak)
    CHnTRK.TRKCH(instanceSize2, nTRK_leak)
    inputlist_leak = []
    for i in paramNameList:
        inputlist_leak.append(widthdict[i])
    global new_finger_dict_leak
    new_finger_dict_leak={}
    new_finger_list_leak1, new_dict_list_leak1=Ecoor.removeFinger_parCombine(instanceSize1, paramNameList, fingnumlist1, inputlist_leak)
    new_finger_list_leak2, new_dict_list_leak2=Ecoor.removeFinger_parCombine(instanceSize2, paramNameList, fingnumlist2, inputlist_leak)
    CHW.widthCH2(instanceSize1, new_dict_list_leak1)
    CHW.widthCH2(instanceSize2, new_dict_list_leak2)
    threads = []
    leakagename = leakageList
    for i in leakagename:
        t = threading.Thread(target = runFineSim, args=(i, ))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    subprocess.call(':grepVar.csh leakagepower {} > leakage1.txt'.format(leakage_lis1), shell=True)
    subprocess.call(':grepVar.csh leakagepower {} > leakage2.txt'.format(leakage_lis2), shell=True)
    subprocess.call(':grepVar.csh leakagepower {} > leakage1_c1.txt'.format(leakage_lis1_c1), shell=True)
    subprocess.call(':grepVar.csh leakagepower {} > leakage2_c1.txt'.format(leakage_lis2_c1), shell=True)
    leakage1=0
    leakage2=0
    leakage1_c1=0
    leakage2_c1=0
    with open ("leakage1.txt", "r") as f:
        f1line = f.readlines()
        for line in f1line:
            leakage1 = float(line)
            print("Leakage1: ", leakage1) 
    with open ("leakage2.txt", "r") as f:
        f2line = f.readlines()
        for line in f2line:
            leakage2 = float(line)
            print("Leakage2: ", leakage2) 
            
    with open ("leakage1_c1.txt", "r") as f:
        f1line_c1 = f.readlines()
        for line in f1line_c1:
            leakage1_c1 = float(line)
            print("Leakage1_c1: ", leakage1_c1) 
    with open ("leakage2_c1.txt", "r") as f:
        f2line_c1 = f.readlines()
        for line in f2line_c1:
            leakage2_c1 = float(line)
            print("Leakage2_c1: ", leakage2_c1) 
    predict_leakage=lp.linearPredict(col_base1,leakage1,col_base2,leakage2,columns)
    predict_leakage_c1=lp.linearPredict(col_base1,leakage1_c1,col_base2,leakage2_c1,columns)
    return leakage1, leakage2, predict_leakage, leakage1_c1, leakage2_c1, predict_leakage_c1
    
def runSimulation (inputParamList, mean, sigma, mean_c1, sigma_c1):
    inputlist = []
    for i in paramNameList:
        inputlist.append(inputParamList[i])
    global new_finger_dict
    new_finger_dict={}
    new_finger_list1, new_dict_list1=Ecoor.removeFinger_parCombine(instanceSize1, paramNameList, fingnumlist1, inputlist)
    new_finger_list2, new_dict_list2=Ecoor.removeFinger_parCombine(instanceSize2, paramNameList, fingnumlist2, inputlist)
    for i in range(0, len(new_finger_list1)):
        new_finger_dict.update({paramNameList[i] : new_finger_list1[i]})
    CHW.widthCH2(instanceSize1,new_dict_list1)
    CHW.widthCH2(instanceSize2,new_dict_list2)
    
    with open(runsp,'w') as fw:
        with open(ref,'r') as f:
            fline = f.readlines()
            for line in fline:
                fw.write(line)
                
    with open(runsp2,'w') as fw:
        with open(ref2,'r') as f:
            fline = f.readlines()
            for line in fline:
                fw.write(line)
    
    with open(runsp_Vth0, 'w') as fw:
        with open(ref_Vth0, 'r') as f:
            fline = f.readlines()
            for line in fline:                                               
                if(VosMean.search(line)):
                    line = line.replace("VosCH1" , "%s" )%float(mean)                    
                if(VosSigma.search(line)):
                    line = line.replace("VosCH2" , "%s" )%float(sigma)                    
                fw.write(line)
                
    with open(runsp2_Vth0, 'w') as fw:
        with open(ref2_Vth0, 'r') as f:
            fline = f.readlines()
            for line in fline:
                if(VosMean.search(line)):
                    line = line.replace("VosCH1" , "%s" )%float(mean)                    
                if(VosSigma.search(line)):
                    line = line.replace("VosCH2" , "%s" )%float(sigma)                    
                fw.write(line)  
                
    with open(runsp_c1,'w') as fw:
        with open(ref_c1,'r') as f:
            fline = f.readlines()
            for line in fline:
                fw.write(line)
                
    with open(runsp2_c1,'w') as fw:
        with open(ref2_c1,'r') as f:
            fline = f.readlines()
            for line in fline:
                fw.write(line)
    
    with open(runsp_Vth0_c1, 'w') as fw:
        with open(ref_Vth0_c1, 'r') as f:
            fline = f.readlines()
            for line in fline:                                               
                if(VosMean.search(line)):
                    line = line.replace("VosCH1" , "%s" )%float(mean_c1)                    
                if(VosSigma.search(line)):
                    line = line.replace("VosCH2" , "%s" )%float(sigma_c1)                    
                fw.write(line)
                
    with open(runsp2_Vth0_c1, 'w') as fw:
        with open(ref2_Vth0_c1, 'r') as f:
            fline = f.readlines()
            for line in fline:
                if(VosMean.search(line)):
                    line = line.replace("VosCH1" , "%s" )%float(mean_c1)                    
                if(VosSigma.search(line)):
                    line = line.replace("VosCH2" , "%s" )%float(sigma_c1)                    
                fw.write(line)              
                
    print('## run simulation ##')
        
    threads = []
    spname = spnameList
    # spname = ["cycle_PCH.sp", "cycle_PCH_Vth0.sp"]

    for i in spname:
        t = threading.Thread(target = runFineSim, args=(i, ))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print('## end ##')
    
    subprocess.call(':grepVar.csh operation_power {}.log > operation_power1.txt'.format(lisFile), shell=True)
    subprocess.call(':grepVar.csh mincycle2 {}.log > minCycle1.txt'.format(lisFile), shell=True)
    subprocess.call(':grepVar.csh accesst {}.log > accessT1.txt'.format(lisFile), shell=True)
    subprocess.call(':grepVar.csh writemargin {}.log > writeMargin1.txt'.format(lisFile_Vth0), shell=True)
    subprocess.call(':grepVar.csh sensingmargin {}.log > sensingMargin1.txt'.format(lisFile_Vth0), shell=True)
    
    subprocess.call(':grepVar.csh operation_power {}.log > operation_power2.txt'.format(lisFile2), shell=True)
    subprocess.call(':grepVar.csh mincycle2 {}.log > minCycle2.txt'.format(lisFile2), shell=True)
    subprocess.call(':grepVar.csh accesst {}.log > accessT2.txt'.format(lisFile2), shell=True)
    subprocess.call(':grepVar.csh writemargin {}.log > writeMargin2.txt'.format(lisFile2_Vth0), shell=True)
    subprocess.call(':grepVar.csh sensingmargin {}.log > sensingMargin2.txt'.format(lisFile2_Vth0), shell=True)

    subprocess.call(':grepVar.csh operation_power {}.log > operation_power1_c1.txt'.format(lisFile_c1), shell=True)
    subprocess.call(':grepVar.csh mincycle2 {}.log > minCycle1_c1.txt'.format(lisFile_c1), shell=True)
    subprocess.call(':grepVar.csh accesst {}.log > accessT1_c1.txt'.format(lisFile_c1), shell=True)
    subprocess.call(':grepVar.csh writemargin {}.log > writeMargin1_c1.txt'.format(lisFile_Vth0_c1), shell=True)
    subprocess.call(':grepVar.csh sensingmargin {}.log > sensingMargin1_c1.txt'.format(lisFile_Vth0_c1), shell=True)
    
    subprocess.call(':grepVar.csh operation_power {}.log > operation_power2_c1.txt'.format(lisFile2_c1), shell=True)
    subprocess.call(':grepVar.csh mincycle2 {}.log > minCycle2_c1.txt'.format(lisFile2_c1), shell=True)
    subprocess.call(':grepVar.csh accesst {}.log > accessT2_c1.txt'.format(lisFile2_c1), shell=True)
    subprocess.call(':grepVar.csh writemargin {}.log > writeMargin2_c1.txt'.format(lisFile2_Vth0_c1), shell=True)
    subprocess.call(':grepVar.csh sensingmargin {}.log > sensingMargin2_c1.txt'.format(lisFile2_Vth0_c1), shell=True)


def extractSimResult (inputParamList, mean, sigma, mean_c1,sigma_c1):
   
    WD_width_REV=0
    SA_width_REV=0
    WD_width_REV_c1=0
    SA_width_REV_c1=0
    ##REV
    with open ("writeMargin1.txt", "r") as f: 
        fline = f.readlines()
        for line in fline:
            print("writeMargin1.txt : ", line)
            if line.find('fail') >= 0:
                print('### Set WD width to the initaial Value  ###')
                # inputParamList['WD1.IWD_L.M0'] = initValue['WD1.IWD_L.M0'] # WD_N
                # inputParamList['WD1.IWD_L.M2'] = initValue['WD1.IWD_L.M2'] # WD_P
                
                runSimulation(inputParamList, mean, sigma, mean_c1, sigma_c1) 
                WD_width_REV += 1
                print("WD_width_REV : ", WD_width_REV)
            else:
                pass
    with open ("writeMargin2.txt", "r") as f: 
        fline = f.readlines()
        for line in fline:
            print("writeMargin2.txt : ", line)
            if line.find('fail') >= 0:
                print('### Set WD width to the initaial Value  ###')
                # inputParamList['WD1.IWD_L.M0'] = initValue['WD1.IWD_L.M0'] # WD_N
                # inputParamList['WD1.IWD_L.M2'] = initValue['WD1.IWD_L.M2'] # WD_P
                runSimulation(inputParamList, mean, sigma, mean_c1, sigma_c1) 
                WD_width_REV += 1
                print("WD_width_REV : ", WD_width_REV)
            else:
                pass
    with open ("accessT1.txt", "r") as f: 
        fline = f.readlines()
        for line in fline:
            print("accessT1.txt : ", line)
            if line.find('fail') >= 0:
                print('### Set SA width to the initaial Value  ###')
                #inputParamList['CNTL.I57.M2'] = initValue['CNTL.I57.M2'] # WD_N
                # inputParamList['SA1.MPD_L'] = initValue['SA1.MPD_L'] # WD_P
                mean,sigma,mean_c1,sigma_c1=SAV.getPostSAos(inputParamList)
                runSimulation(inputParamList, mean, sigma, mean_c1, sigma_c1) 
                SA_width_REV += 1
                print("SA_width_REV : ", SA_width_REV)
            else:
                pass
    with open ("accessT2.txt", "r") as f: 
        fline = f.readlines()
        for line in fline:
            print("accessT2.txt : ", line)
            if line.find('fail') >= 0:
                print('### Set SA width to the initaial Value  ###')
                #inputParamList['CNTL.I57.M2'] = initValue['CNTL.I57.M2'] # WD_N
                # inputParamList['SA1.MPD_L'] = initValue['SA1.MPD_L'] # WD_P
                mean,sigma,mean_c1,sigma_c1=SAV.getPostSAos(inputParamList)
                runSimulation(inputParamList, mean, sigma, mean_c1, sigma_c1) 
                SA_width_REV += 1
                print("SA_width_REV : ", SA_width_REV)
            else:
                pass         
    with open ("writeMargin1_c1.txt", "r") as f: 
        fline = f.readlines()
        for line in fline:
            print("writeMargin1_c1.txt : ", line)
            if line.find('fail') >= 0:
                print('### Set WD width to the initaial Value  ###')
                # inputParamList['WD1.IWD_L.M0'] = initValue['WD1.IWD_L.M0'] # WD_N
                # inputParamList['WD1.IWD_L.M2'] = initValue['WD1.IWD_L.M2'] # WD_P
                
                runSimulation(inputParamList, mean, sigma, mean_c1, sigma_c1) 
                WD_width_REV_c1 += 1
                print("WD_width_REV_c1 : ", WD_width_REV_c1)
            else:
                pass
    with open ("writeMargin2_c1.txt", "r") as f: 
        fline = f.readlines()
        for line in fline:
            print("writeMargin2_c1.txt : ", line)
            if line.find('fail') >= 0:
                print('### Set WD width to the initaial Value  ###')
                # inputParamList['WD1.IWD_L.M0'] = initValue['WD1.IWD_L.M0'] # WD_N
                # inputParamList['WD1.IWD_L.M2'] = initValue['WD1.IWD_L.M2'] # WD_P
                runSimulation(inputParamList, mean, sigma, mean_c1, sigma_c1) 
                WD_width_REV_c1 += 1
                print("WD_width_REV_c1 : ", WD_width_REV_c1)
            else:
                pass
    with open ("accessT1_c1.txt", "r") as f: 
        fline = f.readlines()
        for line in fline:
            print("accessT1_c1.txt : ", line)
            if line.find('fail') >= 0:
                print('### Set SA width to the initaial Value  ###')
                #inputParamList['CNTL.I57.M2'] = initValue['CNTL.I57.M2'] # WD_N
                # inputParamList['SA1.MPD_L'] = initValue['SA1.MPD_L'] # WD_P
                mean,sigma,mean_c1,sigma_c1=SAV.getPostSAos(inputParamList)
                runSimulation(inputParamList, mean, sigma, mean_c1, sigma_c1) 
                SA_width_REV_c1 += 1
                print("SA_width_REV_c1 : ", SA_width_REV_c1)
            else:
                pass
    with open ("accessT2_c1.txt", "r") as f: 
        fline = f.readlines()
        for line in fline:
            print("accessT2_c1.txt : ", line)
            if line.find('fail') >= 0:
                print('### Set SA width to the initaial Value  ###')
                #inputParamList['CNTL.I57.M2'] = initValue['CNTL.I57.M2'] # WD_N
                # inputParamList['SA1.MPD_L'] = initValue['SA1.MPD_L'] # WD_P
                mean,sigma,mean_c1,sigma_c1=SAV.getPostSAos(inputParamList)
                runSimulation(inputParamList, mean, sigma, mean_c1, sigma_c1) 
                SA_width_REV_c1 += 1
                print("SA_width_REV_c1 : ", SA_width_REV_c1)
            else:
                pass 
        
    
    writeMargin1, Energy1, cycle_time1, AccessTime1, ReadMargin1 = extract(1)
    writeMargin2, Energy2, cycle_time2, AccessTime2, ReadMargin2 = extract(2)
    writeMargin1_c1, Energy1_c1, cycle_time1_c1, AccessTime1_c1, ReadMargin1_c1 = extract_c1(1)
    writeMargin2_c1, Energy2_c1, cycle_time2_c1, AccessTime2_c1, ReadMargin2_c1 = extract_c1(2)
    
    
    result1 = [Energy1, cycle_time1, writeMargin1, AccessTime1, ReadMargin1]
    result2 = [Energy2, cycle_time2, writeMargin2, AccessTime2, ReadMargin2]
    result1_c1 = [Energy1_c1, cycle_time1_c1, writeMargin1_c1, AccessTime1_c1, ReadMargin1_c1]
    result2_c1 = [Energy2_c1, cycle_time2_c1, writeMargin2_c1, AccessTime2_c1, ReadMargin2_c1]
    return result1, result2, result1_c1, result2_c1


def extract(num):
    WM=0
    operation_power=0
    minCycle=0
    Taccess=0
    sensingMargin=0
    with open ("writeMargin{}.txt".format(num), "r") as f:
        fline = f.readlines()
        for line in fline:
            WM = float(line)
            print("WriteMargin{}: ".format(num), WM)           
                 
    with open ("operation_power{}.txt".format(num), "r") as f:
        fline = f.readlines()
        for line in fline:
            operation_power = float(line)
            print("operation power{}: ".format(num), operation_power)
            
    with open ("minCycle{}.txt".format(num), "r") as f:
        fline = f.readlines()
        for line in fline:
            minCycle = float(line)
            print("minCycle{}: ".format(num), minCycle)
                
    with open ("accessT{}.txt".format(num), "r") as f:
        fline = f.readlines()
        for line in fline:
            Taccess = float(line)
            print("Access time{}: ".format(num), Taccess) 

    with open ("sensingMargin{}.txt".format(num), "r") as f:
        fline = f.readlines()
        for line in fline:
            sensingMargin = float(line)
            print("sensing Margin{}: ".format(num), sensingMargin) 
    return WM, operation_power, minCycle, Taccess, sensingMargin

def extract_c1(num):
    WM=0
    operation_power=0
    minCycle=0
    Taccess=0
    sensingMargin=0
    with open ("writeMargin{}_c1.txt".format(num), "r") as f:
        fline = f.readlines()
        for line in fline:
            WM = float(line)
            print("WriteMargin{}_c1: ".format(num), WM)           
                 
    with open ("operation_power{}_c1.txt".format(num), "r") as f:
        fline = f.readlines()
        for line in fline:
            operation_power = float(line)
            print("operation power{}_c1: ".format(num), operation_power)
            
    with open ("minCycle{}_c1.txt".format(num), "r") as f:
        fline = f.readlines()
        for line in fline:
            minCycle = float(line)
            print("minCycle{}_c1: ".format(num), minCycle)
                
    with open ("accessT{}_c1.txt".format(num), "r") as f:
        fline = f.readlines()
        for line in fline:
            Taccess = float(line)
            print("Access time{}_c1: ".format(num), Taccess) 

    with open ("sensingMargin{}_c1.txt".format(num), "r") as f:
        fline = f.readlines()
        for line in fline:
            sensingMargin = float(line)
            print("sensing Margin{}_c1: ".format(num), sensingMargin) 
    return WM, operation_power, minCycle, Taccess, sensingMargin



##export spice netlist
# os.system("si -batch -command netlist -cdslib /home/jihwan/virtuoso/cds.lib")


##### initial input List1
init1_inputList = []#초기값
for i in paramList:
    init1_inputList.append(bnd[i][1])
init1_inputList[-1] = initTRKnum

##### initial input List2    
init2_inputList = []#초기값
for i in paramList:
    if(bnd[i][1]*4/5>=0.1):
        i2v=bnd[i][1]*4/5
    else:
        i2v=0.1
    init2_inputList.append(i2v)
init2_inputList[-1] = 22

##### initial Dict1
init_input1 = {}
for i, j in zip(paramList, init1_inputList):
    init_input1.update({i : j})
##### initial Dict2
init_input2 = {}
for i, j in zip(paramList, init2_inputList):
    init_input2.update({i : j})

##### initial width of PG and FOOT    
# init1_w_pg = init_input1['SA_PG_width']
# init1_w_foot = init_input1['SA_FOOT_width']
init1_nTRK = round(init_input1['Ntrk'])

# init2_w_pg = init_input2['SA_PG_width']
# init2_w_foot = init_input2['SA_FOOT_width']
init2_nTRK = round(init_input2['Ntrk'])

if (rows<17):
    init1_nTRK = rows
    init2_nTRK = rows
## Find Vos mean, std 1
# subprocess.call("./getSAvos.py {} 30e-9 2000e-9 90e-9 500e-9 30e-9 {} 30e-9 1".format(init1_w_pg, init1_w_foot), shell=True)
# subprocess.call("./TRKBcell_onNumChange.py {} {}".format(instSize, init1_nTRK), shell=True)
if (initlearn):
    if (os.path.isfile('init1_data.txt')==False or os.path.isfile('init2_data.txt')==False):
        sys.exit("there's no init GP file")
    if(len(add_pa)+len(del_pa)!=0):
        with open('init1_data.txt',"r") as ri1:
            ri1line=ri1.readlines()
            with open('init1_data.txt',"w") as wi1:
                ri1line[0]=str(init_input1)+"\n"
                for cg1 in range(0,len(ri1line)):
                    wi1.write(str(ri1line[cg1]))
        with open('init2_data.txt',"r") as ri2:
            ri2line=ri2.readlines()
            with open('init2_data.txt',"w") as wi2:
                ri2line[0]=str(init_input1)+"\n"
                for cg2 in range(0,len(ri2line)):
                    wi2.write(str(ri2line[cg2]))
        with open('init1_data_c1.txt',"r") as ri1:
            ri1line=ri1.readlines()
            with open('init1_data_c1.txt',"w") as wi1:
                ri1line[0]=str(init_input1)+"\n"
                for cg1 in range(0,len(ri1line)):
                    wi1.write(str(ri1line[cg1]))
        with open('init2_data_c1.txt',"r") as ri2:
            ri2line=ri2.readlines()
            with open('init2_data_c1.txt',"w") as wi2:
                ri2line[0]=str(init_input1)+"\n"
                for cg2 in range(0,len(ri2line)):
                    wi2.write(str(ri2line[cg2]))
                #this start    
                   
                    
    with open('init1_data.txt',"r") as i1r:
        iline1=i1r.readlines()
        for ir in range(0,len(iline1)):
            iline1[ir]=iline1[ir].strip()
    with open('init2_data.txt',"r") as i2r:
        iline2=i2r.readlines()
        for irr in range(0,len(iline2)):
            iline2[irr]=(iline2[irr].strip())
    with open('init_leakage_data.txt',"r") as ilr:
        illine=ilr.readlines()
        for il in range(0,len(illine)):
            illine[il]=(illine[il].strip())
    with open('init1_data_c1.txt',"r") as i1r:
        iline1_c1=i1r.readlines()
        for ir in range(0,len(iline1_c1)):
            iline1_c1[ir]=iline1_c1[ir].strip()
    with open('init2_data_c1.txt',"r") as i2r:
        iline2_c1=i2r.readlines()
        for irr in range(0,len(iline2_c1)):
            iline2_c1[irr]=(iline2_c1[irr].strip())
    with open('init_leakage_data_c1.txt',"r") as ilr:
        illine_c1=ilr.readlines()
        for il in range(0,len(illine_c1)):
            illine_c1[il]=(illine_c1[il].strip())
    # init_input1=iline1[0]
    init1_predict_energy=lp.linearPredict(col_base1, float(iline1[1]), col_base2, float(iline1[2]), columns)
    init1_predict_minCycle=lp.linearPredict(col_base1, float(iline1[3]), col_base2, float(iline1[4]), columns)
    init1_predict_WM=lp.linearPredict(col_base1, float(iline1[5]), col_base2, float(iline1[6]), columns)
    init1_predict_Taccess=lp.linearPredict(col_base1, float(iline1[7]), col_base2, float(iline1[8]), columns)
    init1_predict_RM=lp.linearPredict(col_base1, float(iline1[9]), col_base2, float(iline1[10]), columns)
    # init_input2=iline2[0]
    init2_predict_energy=lp.linearPredict(col_base1, float(iline2[1]), col_base2, float(iline2[2]), columns)
    init2_predict_minCycle=lp.linearPredict(col_base1, float(iline2[3]), col_base2, float(iline2[4]), columns)
    init2_predict_WM=lp.linearPredict(col_base1, float(iline2[5]), col_base2, float(iline2[6]), columns)
    init2_predict_Taccess=lp.linearPredict(col_base1, float(iline2[7]), col_base2, float(iline2[8]), columns)
    init2_predict_RM=lp.linearPredict(col_base1, float(iline2[9]), col_base2, float(iline2[10]), columns)
    
    init_leakage1=float(illine[0])
    init_leakage2=float(illine[1])

    # init_input1=iline1[0]_c1
    init1_predict_energy_c1=lp.linearPredict(col_base1, float(iline1_c1[1]), col_base2, float(iline1_c1[2]), columns)
    init1_predict_minCycle_c1=lp.linearPredict(col_base1, float(iline1_c1[3]), col_base2, float(iline1_c1[4]), columns)
    init1_predict_WM_c1=lp.linearPredict(col_base1, float(iline1_c1[5]), col_base2, float(iline1_c1[6]), columns)
    init1_predict_Taccess_c1=lp.linearPredict(col_base1, float(iline1_c1[7]), col_base2, float(iline1_c1[8]), columns)
    init1_predict_RM_c1=lp.linearPredict(col_base1, float(iline1_c1[9]), col_base2, float(iline1_c1[10]), columns)
    # init_input2=iline2[0]
    init2_predict_energy_c1=lp.linearPredict(col_base1, float(iline2_c1[1]), col_base2, float(iline2_c1[2]), columns)
    init2_predict_minCycle_c1=lp.linearPredict(col_base1, float(iline2_c1[3]), col_base2, float(iline2_c1[4]), columns)
    init2_predict_WM_c1=lp.linearPredict(col_base1, float(iline2_c1[5]), col_base2, float(iline2_c1[6]), columns)
    init2_predict_Taccess_c1=lp.linearPredict(col_base1, float(iline2_c1[7]), col_base2, float(iline2_c1[8]), columns)
    init2_predict_RM_c1=lp.linearPredict(col_base1, float(iline2_c1[9]), col_base2, float(iline2_c1[10]), columns)
    
    init_leakage1_c1=float(illine_c1[0])
    init_leakage2_c1=float(illine_c1[1])
    # init_leakage=float(illine[2])
    init_leakage=lp.linearPredict(col_base1, init_leakage1, col_base2, init_leakage2, columns)
    init_leakage_c1=lp.linearPredict(col_base1, init_leakage1_c1, col_base2, init_leakage2_c1, columns)
    init_yn=1
    tcycleConst = init1_predict_minCycle
    TaccessConst= init1_predict_Taccess
    tcycleConst_c1 = init1_predict_minCycle_c1
    TaccessConst_c1= init1_predict_Taccess_c1
    
    
    init1_tcycle_margin=(tcycleConst-init1_predict_minCycle)/tcycleConst
    init2_tcycle_margin=(tcycleConst-init2_predict_minCycle)/tcycleConst
    init1_tcycle_margin_c1=(tcycleConst_c1-init1_predict_minCycle_c1)/tcycleConst_c1
    init2_tcycle_margin_c1=(tcycleConst_c1-init2_predict_minCycle_c1)/tcycleConst_c1
    
    init1_Taccess_margin=(TaccessConst-init1_predict_Taccess)/TaccessConst
    init2_Taccess_margin=(TaccessConst-init2_predict_Taccess)/TaccessConst
    init1_Taccess_margin_c1=(TaccessConst_c1-init1_predict_Taccess_c1)/TaccessConst_c1
    init2_Taccess_margin_c1=(TaccessConst_c1-init2_predict_Taccess_c1)/TaccessConst_c1
    
    gpr_list=[[TcycleGPR,0,1],[WMGPR,WMConst,0],[TaccessGPR,0,1],[RMGPR,RMConst,1]]#third value : min=0, max=1
    utility = UtilityFunction(kind="custom_multiConst_ex", kappa=1.96, xi=0, gp_list=gpr_list)
    with open("leakageOPT.log_power", "w") as LOG:
        LOG.write("init leakage\n")
        LOG.write("init leakage 1 : "+ str(init_leakage1) + "\n")
        LOG.write("init leakage 2 : "+ str(init_leakage2) + "\n")
        LOG.write("init predict leakage : "+ str(init_leakage) + "\n")
        LOG.write("init leakage 1 c1 : "+ str(init_leakage1_c1) + "\n")
        LOG.write("init leakage 2 c1 : "+ str(init_leakage2_c1) + "\n")
        LOG.write("init predict leakage c1 : "+ str(init_leakage_c1) + "\n\n\n")
    
    
    origin_energy = init1_predict_energy
    origin_energy_c1 = init1_predict_energy_c1
    origin_energy_mc = math.sqrt(init1_predict_energy**2+init1_predict_energy_c1**2)
    
    
    # if init1_predict_RM > 0.09:
    init2_nTRK = round(rows/2)
    init_input2['Ntrk'] = init2_nTRK
        
    init1_energy_dr_a=((init1_predict_energy-init1_predict_energy)/init1_predict_energy)
    init1_energy_dr_b=((init1_predict_energy_c1-init1_predict_energy_c1)/init1_predict_energy_c1)
    if(init1_energy_dr_a<0 and init1_energy_dr_b<0):
        init1_energy_dr_p=-1
    else:
        init1_energy_dr_p=1
    init1_energy_dr=(init1_energy_dr_a*init1_energy_dr_b)*init1_energy_dr_p
    
    init2_energy_dr_a=((init1_predict_energy-init2_predict_energy)/init1_predict_energy)
    init2_energy_dr_b=((init1_predict_energy_c1-init2_predict_energy_c1)/init1_predict_energy_c1)
    if(init2_energy_dr_a<0 and init2_energy_dr_b<0):
        init2_energy_dr_p=-1
    else:
        init2_energy_dr_p=1
    init2_energy_dr=(init2_energy_dr_a*init2_energy_dr_b)*init2_energy_dr_p
    
    # powerBO.register(params = init_input1, target = -math.sqrt(init1_predict_energy**2+init1_predict_energy_c1**2))
    # powerBO.register(params = init_input2, target = -math.sqrt(init2_predict_energy**2+init2_predict_energy_c1**2))
    powerBO.register(params = init_input1, target = init1_energy_dr)
    powerBO.register(params = init_input2, target = init2_energy_dr)
    #initEnergy =[init1_predict_energy, init2_predict_energy]
    initCycle = [min(init1_tcycle_margin,init1_tcycle_margin_c1), min(init2_tcycle_margin,init2_tcycle_margin_c1)]
    initWM = [max(init1_predict_WM,init1_predict_WM_c1), max(init2_predict_WM,init2_predict_WM_c1)]
    initTaccess = [min(init1_Taccess_margin,init1_Taccess_margin_c1), min(init2_Taccess_margin,init2_Taccess_margin_c1)]
    initRM = [min(init1_predict_RM,init1_predict_RM_c1), min(init2_predict_RM,init2_predict_RM_c1)]
    # For constraint GP model
    [X_input, y_tcycle] = [[init1_inputList, init2_inputList], initCycle]
    [X_input, y_WM] = [[init1_inputList, init2_inputList], initWM]
    [X_input, y_taccess] = [[init1_inputList, init2_inputList], initTaccess]
    [X_input, y_RM] = [[init1_inputList, init2_inputList], initRM]
    TcycleGPR.fit(X_input, y_tcycle)
    WMGPR.fit(X_input, y_WM)
    TaccessGPR.fit(X_input, y_taccess)
    RMGPR.fit(X_input, y_RM)
    
    
    
        
else:
    init1_mean, init1_sigma,init1_mean_c1, init1_sigma_c1 = SAV.getPostSAos(init_input1)
    
    
    
    ##layout import, change and export + pex
    
    # CHL.layoutCHready(instanceSize1)
    # CHL.layoutCH_parCombine(TRCoord1, oldWidth, init1_inputList, instanceSize1)
    # CHL.readyForSim(instanceSize1)
    CHnTRK.TRKCH(instanceSize1, init1_nTRK)
    
    # CHW.widthCH2(instanceSize1,init1_init_input1)
    
    # runSi
    
    
    
    
    # CHL.layoutCHready(instanceSize2)
    # CHL.layoutCH_parCombine(TRCoord2, oldWidth, init1_inputList, instanceSize2)
    # CHL.readyForSim(instanceSize2)
    CHnTRK.TRKCH(instanceSize2, init1_nTRK)
    
    
    
    ##### runFinesim for initial observation (1/2)  & register initial point 1
    runSimulation(init_input1, init1_mean, init1_sigma, init1_mean_c1, init1_sigma_c1)
    init1_result1,init1_result2, init1_result1_c1, init1_result2_c1 = extractSimResult(init_input1, init1_mean, init1_sigma, init1_mean_c1, init1_sigma_c1)
    init1_predict_energy=lp.linearPredict(col_base1,init1_result1[0],col_base2,init1_result2[0],columns)
    init1_predict_minCycle=lp.linearPredict(col_base1, init1_result1[1],col_base2, init1_result2[1], columns)           #  (init1_result1[1]+init1_result2[1])/2
    init1_predict_WM=(init1_result1[2]+init1_result2[2])/2
    init1_predict_Taccess=lp.linearPredict(col_base1,init1_result1[3],col_base2,init1_result2[3],columns)
    init1_predict_RM=lp.linearPredict(col_base1,init1_result1[4],col_base2,init1_result2[4],columns)
    init1_predict_energy_c1=lp.linearPredict(col_base1,init1_result1_c1[0],col_base2,init1_result2_c1[0],columns)
    init1_predict_minCycle_c1=lp.linearPredict(col_base1, init1_result1_c1[1],col_base2, init1_result2_c1[1], columns)           #  (init1_result1[1]+init1_result2[1])/2
    init1_predict_WM_c1=(init1_result1_c1[2]+init1_result2_c1[2])/2
    init1_predict_Taccess_c1=lp.linearPredict(col_base1,init1_result1_c1[3],col_base2,init1_result2_c1[3],columns)
    init1_predict_RM_c1=lp.linearPredict(col_base1,init1_result1_c1[4],col_base2,init1_result2_c1[4],columns)
    with open('init1_data.txt', "w") as ind:
        ind.write(str(init_input1)+"\n")
        ind.write(str(init1_result1[0])+"\n")
        ind.write(str(init1_result2[0])+"\n")
        ind.write(str(init1_result1[1])+"\n")
        ind.write(str(init1_result2[1])+"\n") 
        ind.write(str(init1_result1[2])+"\n") 
        ind.write(str(init1_result2[2])+"\n") 
        ind.write(str(init1_result1[3])+"\n") 
        ind.write(str(init1_result2[3])+"\n") 
        ind.write(str(init1_result1[4])+"\n")
        ind.write(str(init1_result2[4])+"\n")
    with open('init1_data_c1.txt', "w") as ind:
        ind.write(str(init_input1)+"\n")
        ind.write(str(init1_result1_c1[0])+"\n")
        ind.write(str(init1_result2_c1[0])+"\n")
        ind.write(str(init1_result1_c1[1])+"\n")
        ind.write(str(init1_result2_c1[1])+"\n") 
        ind.write(str(init1_result1_c1[2])+"\n") 
        ind.write(str(init1_result2_c1[2])+"\n") 
        ind.write(str(init1_result1_c1[3])+"\n") 
        ind.write(str(init1_result2_c1[3])+"\n") 
        ind.write(str(init1_result1_c1[4])+"\n")
        ind.write(str(init1_result2_c1[4])+"\n")
            
    tcycleConst = init1_predict_minCycle
    TaccessConst= init1_predict_Taccess
    tcycleConst_c1 = init1_predict_minCycle_c1
    TaccessConst_c1= init1_predict_Taccess_c1
    
    gpr_list=[[TcycleGPR,0,1],[WMGPR,WMConst,0],[TaccessGPR,0,1],[RMGPR,RMConst,1]]#third value : min=0, max=1
    utility = UtilityFunction(kind="custom_multiConst_ex", kappa=1.96, xi=0, gp_list=gpr_list)
    init_leakage1, init_leakage2, init_leakage,init_leakage1_c1, init_leakage2_c1, init_leakage_c1=measLeakage(init_input1)
    
    with open("leakageOPT.log_power", "w") as LOG:
        LOG.write("init leakage\n")
        LOG.write("init leakage 1 : "+ str(init_leakage1) + "\n")
        LOG.write("init leakage 2 : "+ str(init_leakage2) + "\n")
        LOG.write("init predict leakage : "+ str(init_leakage) + "\n")
        LOG.write("init leakage 1 c1 : "+ str(init_leakage1_c1) + "\n")
        LOG.write("init leakage 2 c1 : "+ str(init_leakage2_c1) + "\n")
        LOG.write("init predict leakage c1 : "+ str(init_leakage_c1) + "\n\n\n")
    with open("init_leakage_data.txt", "w") as il:
        il.write(str(init_leakage1)+"\n")
        il.write(str(init_leakage2)+"\n")
        il.write(str(init_leakage)+"\n")
    with open("init_leakage_data_c1.txt", "w") as il:
        il.write(str(init_leakage1_c1)+"\n")
        il.write(str(init_leakage2_c1)+"\n")
        il.write(str(init_leakage_c1)+"\n")
    
    #powerBO.register(params = init_input1, target = -math.sqrt(init1_predict_energy**2+init1_predict_energy_c1**2))
    init1_energy_dr_a=((init1_predict_energy-init1_predict_energy)/init1_predict_energy)
    init1_energy_dr_b=((init1_predict_energy_c1-init1_predict_energy_c1)/init1_predict_energy_c1)
    if(init1_energy_dr_a<0 and init1_energy_dr_b<0):
        init1_energy_dr_p=-1
    else:
        init1_energy_dr_p=1
    init1_energy_dr=(init1_energy_dr_a*init1_energy_dr_b)*init1_energy_dr_p
    
    powerBO.register(params = init_input1, target = init1_energy_dr)
    origin_energy = init1_predict_energy
    origin_energy_c1 = init1_predict_energy_c1
    origin_energy_mc = math.sqrt(init1_predict_energy**2+init1_predict_energy_c1**2)
    
    
    init2_mean, init2_sigma,init2_mean_c1,init2_sigma_c1 = SAV.getPostSAos(init_input2)
    
    # if init1_predict_RM > 0.1:
    init2_nTRK = round(rows/2)
    init_input2['Ntrk'] = init2_nTRK
    
    ##layout import, change and export + pex
    # CHL.layoutCHready(instanceSize1)
    # CHL.layoutCH_parCombine(TRCoord1, oldWidth, init2_inputList, instanceSize1)
    # CHL.readyForSim(instanceSize1)
    CHnTRK.TRKCH(instanceSize1, init2_nTRK)
    
    
    
    # CHL.layoutCHready(instanceSize2)
    # CHL.layoutCH_parCombine(TRCoord2, oldWidth, init2_inputList, instanceSize2)
    # CHL.readyForSim(instanceSize2)
    CHnTRK.TRKCH(instanceSize2, init2_nTRK)
    
    ##### runFinesim for initial observation (1/2)  & register initial point 1
    runSimulation(init_input2, init2_mean, init2_sigma, init2_mean_c1, init2_sigma_c1)
    init2_result1,init2_result2,init2_result1_c1,init2_result2_c1 = extractSimResult(init_input2, init2_mean, init2_sigma, init2_mean_c1, init2_sigma_c1)
    init2_predict_energy=lp.linearPredict(col_base1,init2_result1[0],col_base2,init2_result2[0],columns)
    init2_predict_minCycle=lp.linearPredict(col_base1, init2_result1[1],col_base2, init2_result2[1],columns)#(init1_result1[1]+init1_result2[1])/2
    init2_predict_WM=(init2_result1[2]+init2_result2[2])/2
    init2_predict_Taccess=lp.linearPredict(col_base1,init2_result1[3],col_base2,init2_result2[3],columns)
    init2_predict_RM=lp.linearPredict(col_base1,init2_result1[4],col_base2,init2_result2[4],columns)
    init2_predict_energy_c1=lp.linearPredict(col_base1,init2_result1_c1[0],col_base2,init2_result2_c1[0],columns)
    init2_predict_minCycle_c1=lp.linearPredict(col_base1, init2_result1_c1[1],col_base2, init2_result2_c1[1],columns)#(init1_result1[1]+init1_result2[1])/2
    init2_predict_WM_c1=(init2_result1_c1[2]+init2_result2_c1[2])/2
    init2_predict_Taccess_c1=lp.linearPredict(col_base1,init2_result1_c1[3],col_base2,init2_result2_c1[3],columns)
    init2_predict_RM_c1=lp.linearPredict(col_base1,init2_result1_c1[4],col_base2,init2_result2_c1[4],columns)
    with open('init2_data.txt', "w") as ind:
            ind.write(str(init_input2)+"\n")
            ind.write(str(init2_result1[0])+"\n")
            ind.write(str(init2_result2[0])+"\n")
            ind.write(str(init2_result1[1])+"\n")
            ind.write(str(init2_result2[1])+"\n") 
            ind.write(str(init2_result1[2])+"\n") 
            ind.write(str(init2_result2[2])+"\n") 
            ind.write(str(init2_result1[3])+"\n") 
            ind.write(str(init2_result2[3])+"\n") 
            ind.write(str(init2_result1[4])+"\n")
            ind.write(str(init2_result2[4])+"\n")
    with open('init2_data_c1.txt', "w") as ind:
            ind.write(str(init_input2)+"\n")
            ind.write(str(init2_result1_c1[0])+"\n")
            ind.write(str(init2_result2_c1[0])+"\n")
            ind.write(str(init2_result1_c1[1])+"\n")
            ind.write(str(init2_result2_c1[1])+"\n") 
            ind.write(str(init2_result1_c1[2])+"\n") 
            ind.write(str(init2_result2_c1[2])+"\n") 
            ind.write(str(init2_result1_c1[3])+"\n") 
            ind.write(str(init2_result2_c1[3])+"\n") 
            ind.write(str(init2_result1_c1[4])+"\n")
            ind.write(str(init2_result2_c1[4])+"\n")
            
    #powerBO.register(params = init_input2, target = -math.sqrt(init2_predict_energy**2+init2_predict_energy_c1**2))#dict
    
    init2_energy_dr_a=((init1_predict_energy-init2_predict_energy)/init1_predict_energy)
    init2_energy_dr_b=((init1_predict_energy_c1-init2_predict_energy_c1)/init1_predict_energy_c1)
    if(init2_energy_dr_a<0 and init2_energy_dr_b<0):
        init2_energy_dr_p=-1
    else:
        init2_energy_dr_p=1
    init2_energy_dr=(init2_energy_dr_a*init2_energy_dr_b)*init2_energy_dr_p
    
    powerBO.register(params = init_input2, target = init2_energy_dr)
    
    init1_tcycle_margin=(tcycleConst-init1_predict_minCycle)/tcycleConst
    init2_tcycle_margin=(tcycleConst-init2_predict_minCycle)/tcycleConst
    init1_tcycle_margin_c1=(tcycleConst_c1-init1_predict_minCycle_c1)/tcycleConst_c1
    init2_tcycle_margin_c1=(tcycleConst_c1-init2_predict_minCycle_c1)/tcycleConst_c1
    
    init1_Taccess_margin=(TaccessConst-init1_predict_Taccess)/TaccessConst
    init2_Taccess_margin=(TaccessConst-init2_predict_Taccess)/TaccessConst
    init1_Taccess_margin_c1=(TaccessConst_c1-init1_predict_Taccess_c1)/TaccessConst_c1
    init2_Taccess_margin_c1=(TaccessConst_c1-init2_predict_Taccess_c1)/TaccessConst_c1
    
    # initEnergy =[init1_predict_energy, init2_predict_energy]
    initCycle = [min(init1_tcycle_margin,init1_tcycle_margin_c1), min(init2_tcycle_margin,init2_tcycle_margin_c1)]
    initWM = [max(init1_predict_WM,init1_predict_WM_c1), max(init2_predict_WM,init2_predict_WM_c1)]
    initTaccess = [min(init1_Taccess_margin,init1_Taccess_margin_c1), min(init2_Taccess_margin,init2_Taccess_margin_c1)]
    initRM = [min(init1_predict_RM,init1_predict_RM_c1), min(init2_predict_RM,init2_predict_RM_c1)]
    # For constraint GP model
    [X_input, y_tcycle] = [[init1_inputList, init2_inputList], initCycle]
    [X_input, y_WM] = [[init1_inputList, init2_inputList], initWM]
    [X_input, y_taccess] = [[init1_inputList, init2_inputList], initTaccess]
    [X_input, y_RM] = [[init1_inputList, init2_inputList], initRM]
    TcycleGPR.fit(X_input, y_tcycle)
    WMGPR.fit(X_input, y_WM)
    TaccessGPR.fit(X_input, y_taccess)
    RMGPR.fit(X_input, y_RM)
# if(LRmodel):
#     PowerLR.fit(X_input,initEnergy)
#     WMLR.fit(X_input,y_WM)
#     TaccessLR.fit(X_input,y_taccess)
#     RMLR.fit(X_input,y_RM)
#     TcycleLR.fit(X_input,y_tcycle)
# iter data load
pre_iter=0
pa_ch=0
energyInConstraint_list_old=[]
energyInConstraint_list_old1=[]
energyInConstraint_list_old2=[]
mincycleInConstraint_list_old = []
mincycleInConstraint_list_old1 = []
mincycleInConstraint_list_old2 = []
WMInConstraint_list_old = []
WMInConstraint_list_old1 = []
WMInConstraint_list_old2 = []
TaccessInConstraint_list_old = []
TaccessInConstraint_list_old1 = []
TaccessInConstraint_list_old2 = []
RMInConstraint_list_old = []
RMInConstraint_list_old1 = []
RMInConstraint_list_old2 = []

param_list_old=[]
inputdict_list_old=[]
#%%
pre_no=1
if (GPlearn):
    if not ((os.path.isfile('ParamName_data.txt')) and(os.path.isfile('Input_data.txt')) and (os.path.isfile('Energy_data1.txt')) and
    (os.path.isfile('minCycle_data1.txt')) and (os.path.isfile('Taccess_data1.txt')) and (os.path.isfile('WM_data1.txt') and (os.path.isfile('RM_data1.txt')))):
        sys.exit("no GP load file")
    with open('ParamName_data.txt',"r") as pnr:
        pline=pnr.readlines()
        for pr in range(0,len(pline)):
            pline[pr]=pline[pr].strip()
        pline_dn=copy.deepcopy(pline)
        pline_dn.remove('Ntrk')
    if (sorted(paramNameList)!=sorted(userParamList)):
        pa_ch=1
    with open('Input_data.txt',"r") as inr:
         inline_f=[]
         inline_v=[]
         inrline=[]
         inline_n={}
         inrline=inr.readlines()
         for ir in range(len(inrline)):
             inrdict=eval(inrline[ir])
             inline_f.append(inrdict)
         if (len(add_pa)!=0):
             for l in range(len(inline_f)):
                 for ap in range(0,len(add_pa)):
                            inline_f[l].update({add_pa[ap]:initValue[add_pa[ap]]})
             with open ('Input_data.txt', "w") as inw:
                 for iw in range(0,len(inline_f)):
                     inline_f_key=(sorted(inline_f[iw]))
                     for nn in range(0,len(inline_f_key)):
                         inline_n.update({inline_f_key[nn]:inline_f[iw][inline_f_key[nn]]})
                     inw.write(str(inline_n)+"\n")
         for iir in range(len(inrline)):
             inline_v.append(list(inline_f[iir].values()))
         pre_iter_num=len(inline_f)
    with open('Energy_data1.txt',"r") as er:
        e1line=er.readlines()
        for ee in range(0,len(e1line)):
            e1line[ee]=e1line[ee].strip()
        eline_f1=[]
        for e in range(0,len(e1line)):
            eline_f1.append(float(e1line[e]))
    with open('minCycle_data1.txt',"r") as cr:
        cline1=cr.readlines()
        for cc in range(0,len(cline1)):
            cline1[cc]=float(cline1[cc].strip())
    with open('Taccess_data1.txt',"r") as tr:
        tline1=tr.readlines()
        for tt in range(0,len(tline1)):
            tline1[tt]=float(tline1[tt].strip())
    with open('WM_data1.txt',"r") as wr:
        wline1=wr.readlines()
        for ww in range(0,len(wline1)):
            wline1[ww]=float(wline1[ww].strip())
    with open('RM_data1.txt',"r") as rr:
        rline1=rr.readlines()
        for rr in range(0,len(rline1)):
            rline1[rr]=float(rline1[rr].strip())
            
    with open('Energy_data2.txt',"r") as er:
        e2line=er.readlines()
        for ee in range(0,len(e2line)):
            e2line[ee]=e2line[ee].strip()
        eline_f2=[]
        for e in range(0,len(e2line)):
            eline_f2.append(float(e2line[e]))
    with open('minCycle_data2.txt',"r") as cr:
        cline2=cr.readlines()
        for cc in range(0,len(cline2)):
            cline2[cc]=float(cline2[cc].strip())
    with open('Taccess_data2.txt',"r") as tr:
        tline2=tr.readlines()
        for tt in range(0,len(tline2)):
            tline2[tt]=float(tline2[tt].strip())
    with open('WM_data2.txt',"r") as wr:
        wline2=wr.readlines()
        for ww in range(0,len(wline2)):
            wline2[ww]=float(wline2[ww].strip())
    with open('RM_data2.txt',"r") as rr:
        rline2=rr.readlines()
        for rr in range(0,len(rline2)):
            rline2[rr]=float(rline2[rr].strip())            
    
    eline_k=[]
    eline_f=[]
    cline=[]
    wline=[]
    tline=[]
    rline=[]
    
    for pre_iter in range(0, len(eline_f1)):
        eline_k=lp.linearPredict(col_base1,eline_f1[pre_iter],col_base2,eline_f2[pre_iter],columns)
        eline_f.append(eline_k)
        cline.append(lp.linearPredict(col_base1,cline1[pre_iter],col_base2,cline2[pre_iter],columns))#(init1_result1[1]+init1_result2[1])/2
        wline.append((wline1[pre_iter]+wline2[pre_iter])/2)
        tline.append(lp.linearPredict(col_base1,tline1[pre_iter],col_base2,tline2[pre_iter],columns))
        rline.append(lp.linearPredict(col_base1,rline1[pre_iter],col_base2,rline2[pre_iter],columns))
        powerBO.register(params = inline_f[pre_iter], target = -1*eline_k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TcycleGPR.fit(inline_v, cline)
        WMGPR.fit(inline_v, wline)
        TaccessGPR.fit(inline_v, tline)
        RMGPR.fit(inline_v, rline)
    if(LRmodel):
        PowerLR.fit(inline_v,eline_f)
        WMLR.fit(inline_v,wline)
        TaccessLR.fit(inline_v,tline)
        RMLR.fit(inline_v,rline)
        TcycleLR.fit(inline_v,cline)
    pre_iter=1
    
    for oldr in range(0,len(cline)):
        if(cline[oldr] < tcycleConst and wline[oldr] < WMConst and tline[oldr] < TaccessConst and rline[oldr] > RMConst):
            inputdict_list_old.append(inline_f[oldr])
            energyInConstraint_list_old.append(float(eline_f[oldr]))
            energyInConstraint_list_old1.append(float(eline_f1[oldr]))
            energyInConstraint_list_old2.append(float(eline_f2[oldr]))
            mincycleInConstraint_list_old.append(float(cline[oldr]))
            mincycleInConstraint_list_old1.append(float(cline1[oldr]))
            mincycleInConstraint_list_old2.append(float(cline2[oldr]))
            WMInConstraint_list_old.append(float(wline[oldr]))
            WMInConstraint_list_old1.append(float(wline1[oldr]))
            WMInConstraint_list_old2.append(float(wline2[oldr]))
            TaccessInConstraint_list_old.append(float(tline[oldr]))
            TaccessInConstraint_list_old1.append(float(tline1[oldr]))
            TaccessInConstraint_list_old2.append(float(tline2[oldr]))
            RMInConstraint_list_old.append(float(rline[oldr]))
            RMInConstraint_list_old1.append(float(rline1[oldr]))
            RMInConstraint_list_old2.append(float(rline2[oldr]))
            pre_no=0
    if (pre_no==0):
        min_index=energyInConstraint_list_old.index(min(energyInConstraint_list_old))
        min_energy_old=min(energyInConstraint_list_old)
        min_energy_old1=energyInConstraint_list_old1[min_index]
        min_energy_old2=energyInConstraint_list_old2[min_index]
        min_tcycle_old=mincycleInConstraint_list_old[min_index]
        min_tcycle_old1=mincycleInConstraint_list_old1[min_index]
        min_tcycle_old2=mincycleInConstraint_list_old2[min_index]
        min_wm_old=WMInConstraint_list_old[min_index]
        min_wm_old1=WMInConstraint_list_old1[min_index]
        min_wm_old2=WMInConstraint_list_old2[min_index]
        min_taccess_old=TaccessInConstraint_list_old[min_index]
        min_taccess_old1=TaccessInConstraint_list_old1[min_index]
        min_taccess_old2=TaccessInConstraint_list_old2[min_index]
        min_rm_old=RMInConstraint_list_old[min_index]
        min_rm_old1=RMInConstraint_list_old1[min_index]
        min_rm_old2=RMInConstraint_list_old2[min_index]
        min_param=inputdict_list_old[min_index]
        min_const={'minCycle' : min_tcycle_old, 'WM' : min_wm_old, 'Taccess' : min_taccess_old, 'RM' : min_rm_old}
        min_const1={'minCycle' : min_tcycle_old1, 'WM' : min_wm_old1, 'Taccess' : min_taccess_old1, 'RM' : min_rm_old1}
        min_const2={'minCycle' : min_tcycle_old2, 'WM' : min_wm_old2, 'Taccess' : min_taccess_old2, 'RM' : min_rm_old2}
    elif (pre_no==1):
        min_energy_old=init1_predict_energy
        min_tcycle_old=init1_predict_minCycle
        min_wm_old=init1_predict_WM
        min_taccess_old=init1_predict_Taccess
        min_rm_old=init1_predict_RM
        min_param=init_input1
        min_const={'minCycle' : min_tcycle_old, 'WM' : min_wm_old, 'Taccess' : min_taccess_old, 'RM' : min_rm_old}
    
    
else:
    min_energy_old=init1_predict_energy
    min_tcycle_old=init1_predict_minCycle
    min_wm_old=init1_predict_WM
    min_taccess_old=init1_predict_Taccess
    min_rm_old=init1_predict_RM
    min_const={'minCycle' : min_tcycle_old, 'WM' : min_wm_old, 'Taccess' : min_taccess_old, 'RM' : min_rm_old}
    min_energy_old_c1=init1_predict_energy_c1
    min_tcycle_old_c1=init1_predict_minCycle_c1
    min_wm_old_c1=init1_predict_WM_c1
    min_taccess_old_c1=init1_predict_Taccess_c1
    min_rm_old_c1=init1_predict_RM_c1
    min_param=init_input1
    min_const_c1={'minCycle' : min_tcycle_old_c1, 'WM' : min_wm_old_c1, 'Taccess' : min_taccess_old_c1, 'RM' : min_rm_old_c1}





# # init value for plot
# initEnergy =[init1_predict_energy, init2_predict_energy]
# initCycle = [init1_predict_minCycle, init2_predict_minCycle]
# initWM = [init1_predict_WM, init2_predict_WM]
# initTaccess = [init1_predict_Taccess, init2_predict_Taccess]
# initRM = [init1_predict_RM, init2_predict_RM]
# # For constraint GP model
# [X_input, y_tcycle] = [[init1_inputList, init2_inputList], initCycle]
# [X_input, y_WM] = [[init1_inputList, init2_inputList], initWM]
# [X_input, y_taccess] = [[init1_inputList, init2_inputList], initTaccess]
# [X_input, y_RM] = [[init1_inputList, init2_inputList], initRM]
# TcycleGPR.fit(X_input, y_tcycle)
# WMGPR.fit(X_input, y_WM)
# TaccessGPR.fit(X_input, y_taccess)
# RMGPR.fit(X_input, y_RM)


if(len(energyInConstraint_list_old)!=0):
    #target_e=-min(eline_f)#energyInConstraint_list_old this????
    target_e=-min(energyInConstraint_list_old)#energyInConstraint_list_old this????
    bo_iter = bo_pre_iter
else:
    #target_e=-min(math.sqrt(init1_predict_energy**2+init1_predict_energy_c1**2), math.sqrt(init2_predict_energy**2+init2_predict_energy_c1**2))
    target_e=max(init1_energy_dr,init2_energy_dr)
param_list = [] 
param_list.append(X_input)
######
np.seterr(invalid="ignore")
next_point = powerBO.suggest(utility, y_max=target_e)
# w_pg = next_point['SA_PG_width']
# w_foot = next_point['SA_FOOT_width']
nTRK = next_point['Ntrk']
energyInConstraint_list=[min_energy_old]
mincycleInConstraint_list = []
WMInConstraint_list = []
TaccessInConstraint_list = []
RMInConstraint_list = []
energyInConstraint_list_c1=[min_energy_old_c1]
mincycleInConstraint_list_c1 = []
WMInConstraint_list_c1 = []
TaccessInConstraint_list_c1 = []
RMInConstraint_list_c1 = []
drInConstraint_list=[init1_energy_dr]
## 
powerGP_list = []
TcycleGPR_list = []
WMGPR_list = []
TaccessGPR_list = []
RMGPR_list = []

Energy_list = []
Energy_list_c1=[]
DR_list=[]
#constraint list
minCycle_list = []
WM_list = []
Taccess_list = []
RM_list = []
minCycle_list_c1 = []
WM_list_c1 = []
Taccess_list_c1 = []
RM_list_c1 = []

param_dict = {}
const_dict = {}
param_dict_1={}
param_dict_2={}
const_dict_1={}
const_dict_2={}
param_finger = {}
param_fingerConsider = {}
param_Nfinger = {}
mean_list = []
sigma_list = []
mean_list_c1 = []
sigma_list_c1 = []

best_energy_list1 = []
best_energy_list1_c1 = []
best_energy_list2 = []
k = 0
with open("powerOPT.log_{}".format(today), "w") as LOG:
    LOG.write("init spec\n")
    LOG.write("energy : "+ str(init1_predict_energy) + "\n")
    LOG.write("Tcycle : "+ str(init1_predict_minCycle) + "\n")
    LOG.write("Taccess : "+ str(init1_predict_Taccess) + "\n")
    LOG.write("WM : "+ str(init1_predict_WM )+ "\n")
    LOG.write("RM : "+ str(init1_predict_RM )+ "\n")
    LOG.write("energy_c1 : "+ str(init1_predict_energy_c1) + "\n")
    LOG.write("Tcycle_c1 : "+ str(init1_predict_minCycle_c1) + "\n")
    LOG.write("Taccess_c1 : "+ str(init1_predict_Taccess_c1) + "\n")
    LOG.write("WM_c1 : "+ str(init1_predict_WM_c1 )+ "\n")
    LOG.write("RM_c1 : "+ str(init1_predict_RM_c1 )+ "\n\n\n")
    LOG.write("spec changing\n")
#%%
print("iteration start")
opt=False
while True:
    k +=1
    # print(str(k)+"th iteration")
    current_point = next_point

    WD_width_REV=0
    SA_width_REV=0
    WD_width_REV_c1=0
    SA_width_REV_c1=0
    # =============================================================================
#     
# =============================================================================
    print("\nSuggested design (current_point): ")
    if ('SA1.MPU_L' in userParamList) and ('SA1.MPG_L' in userParamList) and ('SA1.MSAPCH_L' in userParamList):
        current_point['SA1.MPU_L'] = current_point['SA1.MPG_L']
        current_point['SA1.MSAPCH_L'] = current_point['SA1.MPG_L']
    print(current_point)

# =============================================================================
# 
# =============================================================================
    currentInputList = []
    for i in paramList:
        currentInputList.append(current_point[i])
    nTRK = round(current_point['Ntrk'])
    # subprocess.call("./getSAvos.py {} 30e-9 2000e-9 90e-9 500e-9 30e-9 {} 30e-9 1".format(w_pg, w_foot), shell=True)
    # subprocess.call("./TRKBcell_onNumChange.py {} {}".format(instSize, nTRK), shell=True)
    # f = open('./Vos.txt', 'r')
    # lines = f.readlines()
    # b = lines[0].split()
    # mean = -float(b[1][:-1])
    # sigma = -float(b[3][:-2])
    # f.close()
    if(LRmodel):
        print("\n####### iteration count : {} ########".format(k))
        new_finger_dict={}
        lr_input=list(current_point.values())
        new_finger_list1, new_dict_list1=Ecoor.removeFinger_parCombine(instanceSize1, paramNameList, fingnumlist1, lr_input)
        for i in range(0, len(new_finger_list1)):
            new_finger_dict.update({paramNameList[i] : new_finger_list1[i]})
        predict_energy = PowerLR.predict([lr_input])[0]
        predict_minCycle = TcycleLR.predict([lr_input])[0] #(minCycleBase1 + minCycleBase2)/2
        predict_Taccess = TaccessLR.predict([lr_input])[0]
        predict_RM = RMLR.predict([lr_input])[0]
        predict_WM = WMLR.predict([lr_input])[0]
    else:
        mean, sigma,mean_c1,sigma_c1 = SAV.getPostSAos(current_point)
        # mean = 0.000144799051832427 #-float(b[1][:-1])
        # sigma = 0.00491978824373201 #-float(b[3][:-2])
        
        # if (k%pex_netlist_count == 0):
        #     CHL.layoutCHready(instSize)
        #     CHL.layoutCH_parCombine(TRCoord, oldWidth, currentInputList)
        #     CHL.readyForSim(instSize)
        # else:
            # subprocess.call("findandchangeWidth.py "+instSize+", "+currentInputList)
        mean_list.append(mean)
        sigma_list.append(sigma)
        mean_list_c1.append(mean_c1)
        sigma_list_c1.append(sigma_c1)

        current_point_copy=copy.deepcopy(current_point)
    
        
        CHnTRK.TRKCH(instanceSize1, nTRK)
        CHnTRK.TRKCH(instanceSize2, nTRK)
        runSimulation(current_point,mean,sigma,mean_c1,sigma_c1)
        
        
        
        
        print("\n####### iteration count : {} ########".format(k))
        
        result1, result2, result1_c1, result2_c1 = extractSimResult(current_point, mean, sigma,mean_c1,sigma_c1)
        energyBase1 = result1[0]
        minCycleBase1 = result1[1]
        WMBase1 = result1[2]
        TaccessBase1 = result1[3]
        RMBase1= result1[4]
        
        energyBase2 = result2[0]
        minCycleBase2 = result2[1]
        WMBase2 = result2[2]
        TaccessBase2 = result2[3]
        RMBase2= result2[4]
    
        predict_energy = lp.linearPredict(col_base1, energyBase1, col_base2, energyBase2, col_target)
        predict_minCycle = lp.linearPredict(col_base1, minCycleBase1,col_base2, minCycleBase2,col_target) #(minCycleBase1 + minCycleBase2)/2
        predict_Taccess = lp.linearPredict(col_base1, TaccessBase1, col_base2,TaccessBase2 , col_target)
        predict_RM = lp.linearPredict(col_base1, RMBase1, col_base2, RMBase2, col_target)
        predict_WM = (WMBase1 + WMBase2)/2
        
        energyBase1_c1 = result1_c1[0]
        minCycleBase1_c1 = result1_c1[1]
        WMBase1_c1 = result1_c1[2]
        TaccessBase1_c1 = result1_c1[3]
        RMBase1_c1= result1_c1[4]
        
        energyBase2_c1 = result2_c1[0]
        minCycleBase2_c1 = result2_c1[1]
        WMBase2_c1 = result2_c1[2]
        TaccessBase2_c1 = result2_c1[3]
        RMBase2_c1= result2_c1[4]
    
        predict_energy_c1 = lp.linearPredict(col_base1, energyBase1_c1, col_base2, energyBase2_c1, col_target)
        predict_minCycle_c1 = lp.linearPredict(col_base1, minCycleBase1_c1,col_base2, minCycleBase2_c1,col_target) #(minCycleBase1 + minCycleBase2)/2
        predict_Taccess_c1 = lp.linearPredict(col_base1, TaccessBase1_c1, col_base2,TaccessBase2_c1 , col_target)
        predict_RM_c1 = lp.linearPredict(col_base1, RMBase1_c1, col_base2, RMBase2_c1, col_target)
        predict_WM_c1 = (WMBase1_c1 + WMBase2_c1)/2
    
    predict_tcycle_margin=(tcycleConst-predict_minCycle)/tcycleConst
    predict_tcycle_margin_c1=(tcycleConst_c1-predict_minCycle_c1)/tcycleConst_c1
    
    predict_Taccess_margin=(TaccessConst-predict_Taccess)/TaccessConst
    predict_Taccess_margin_c1=(TaccessConst_c1-predict_Taccess_c1)/TaccessConst_c1
    
    
    print('#### predict energy & constraint ####')
    print('predict energy : ', predict_energy)
    print('predict minCycle : ', predict_minCycle)
    print('predict Taccess : ', predict_Taccess)
    print('predict RM : ', predict_RM)
    print('predict WM : ', predict_WM)
    print('predict energy_c1 : ', predict_energy_c1)
    print('predict minCycle_c1 : ', predict_minCycle_c1)
    print('predict Taccess_c1 : ', predict_Taccess_c1)
    print('predict RM_c1 : ', predict_RM_c1)
    print('predict WM_c1 : ', predict_WM_c1)
    print('predict tcycle margin : ', predict_tcycle_margin)
    print('predict taccess margin : ', predict_Taccess_margin)
    print('predict tcycle margin_c1 : ', predict_tcycle_margin_c1)
    print('predict taccess margin_c1 : ', predict_Taccess_margin_c1)
    print('minCycle Const :', tcycleConst)
    print('Taccess Const :', TaccessConst)
    predict_energy_dr_a=((init1_predict_energy-predict_energy)/init1_predict_energy)
    predict_energy_dr_b=((init1_predict_energy_c1-predict_energy_c1)/init1_predict_energy_c1)
    if(predict_energy_dr_a<0 and predict_energy_dr_b<0):
        predict_energy_dr_p=-1
    else:
        predict_energy_dr_p=1
    predict_energy_dr=(predict_energy_dr_a*predict_energy_dr_b)*predict_energy_dr_p
    
    #loss_function = math.sqrt(predict_energy**2+predict_energy_c1**2)
    loss_function=predict_energy_dr
    
    
    Energy_list.append(predict_energy)
    minCycle_list.append(predict_minCycle)
    Taccess_list.append(predict_Taccess)
    WM_list.append(predict_WM)
    RM_list.append(predict_RM)
    Energy_list_c1.append(predict_energy_c1)
    minCycle_list_c1.append(predict_minCycle_c1)
    Taccess_list_c1.append(predict_Taccess_c1)
    WM_list_c1.append(predict_WM_c1)
    RM_list_c1.append(predict_RM_c1)
    DR_list.append(predict_energy_dr)
    
    # if not (LRmodel):
    with open('Input_data.txt', "a") as ind:
        ind.write(str(current_point)+"\n")    
        
    with open('Energy_data1.txt', "a") as ed:
        ed.write(str(energyBase1)+"\n")
    with open('minCycle_data1.txt', "a") as cd:
        cd.write(str(minCycleBase1)+"\n")
    with open('Taccess_data1.txt', "a") as td:
        td.write(str(TaccessBase1)+"\n")
    with open('WM_data1.txt', "a") as wd:
        wd.write(str(WMBase1)+"\n")
    with open('RM_data1.txt', "a") as rd:
        rd.write(str(RMBase1)+"\n")
        
    with open('Energy_data2.txt', "a") as ed:
        ed.write(str(energyBase2)+"\n")
    with open('minCycle_data2.txt', "a") as cd:
        cd.write(str(minCycleBase2)+"\n")
    with open('Taccess_data2.txt', "a") as td:
        td.write(str(TaccessBase2)+"\n")
    with open('WM_data2.txt', "a") as wd:
        wd.write(str(WMBase2)+"\n")
    with open('RM_data2.txt', "a") as rd:
        rd.write(str(RMBase2)+"\n")
    
    with open('Energy_data1_c1.txt', "a") as ed:
        ed.write(str(energyBase1_c1)+"\n")
    with open('minCycle_data1_c1.txt', "a") as cd:
        cd.write(str(minCycleBase1_c1)+"\n")
    with open('Taccess_data1_c1.txt', "a") as td:
        td.write(str(TaccessBase1_c1)+"\n")
    with open('WM_data1_c1.txt', "a") as wd:
        wd.write(str(WMBase1_c1)+"\n")
    with open('RM_data1_c1.txt', "a") as rd:
        rd.write(str(RMBase1_c1)+"\n")
        
    with open('Energy_data2_c1.txt', "a") as ed:
        ed.write(str(energyBase2_c1)+"\n")
    with open('minCycle_data2_c1.txt', "a") as cd:
        cd.write(str(minCycleBase2_c1)+"\n")
    with open('Taccess_data2_c1.txt', "a") as td:
        td.write(str(TaccessBase2_c1)+"\n")
    with open('WM_data2_c1.txt', "a") as wd:
        wd.write(str(WMBase2_c1)+"\n")
    with open('RM_data2_c1.txt', "a") as rd:
        rd.write(str(RMBase2_c1)+"\n")
    
    if(k==1):
        with open('ParamName_data.txt', "w") as pn:
            for p in range(0,len(list(current_point.keys()))):
                pn.write(list(current_point.keys())[p]+'\n')

    
    
    
    
    print("WD_width_REV : ", WD_width_REV)
    print("SA_width_REV : ", SA_width_REV)
    print("WD_width_REV_c1 : ", WD_width_REV_c1)
    print("SA_width_REV_c1 : ", SA_width_REV_c1)
    # if WD_width_REV ==1:
    #     a9 = initValue[param9]
    #     a10 = initValue[param10]
    # else:
    #     pass
    
    if(predict_energy_dr>0 and min(predict_tcycle_margin,predict_tcycle_margin_c1) >0 and min(predict_Taccess_margin,predict_Taccess_margin_c1) >0 and max(predict_WM,predict_WM_c1) < WMConst and min(predict_RM,predict_RM_c1) > RMConst):
        energyInConstraint_list.append(predict_energy)
        mincycleInConstraint_list.append(predict_minCycle)
        TaccessInConstraint_list.append(predict_Taccess)
        WMInConstraint_list.append(predict_WM)
        RMInConstraint_list.append(predict_RM)
        energyInConstraint_list_c1.append(predict_energy_c1)
        mincycleInConstraint_list_c1.append(predict_minCycle_c1)
        TaccessInConstraint_list_c1.append(predict_Taccess_c1)
        WMInConstraint_list_c1.append(predict_WM_c1)
        RMInConstraint_list_c1.append(predict_RM_c1)
        drInConstraint_list.append(predict_energy_dr)
        # for j in paramList:
            # param_dict.update({"{}".format(Energy) : {param+'{}'.format(i) : current_point["{}".format(j)]}})
        param_dict.update({"{}".format(math.sqrt(predict_energy**2+predict_energy_c1**2)) : current_point}) ##REV
        # param_dict_1.update({"{}".format(predict_energy) : energyBase1})
        # param_dict_2.update({"{}".format(predict_energy) : energyBase2})
        param_finger.update({math.sqrt(predict_energy**2+predict_energy_c1**2) : new_finger_dict})
        param_cosFin = {}
        param_Nfin = {}
        for new_param_key, new_param_value in list(param_dict.values())[-1].items():
            if (new_param_key == 'Ntrk'):
                continue
            param_cosFin.update({new_param_key : new_param_value/new_finger_dict[new_param_key]})
            param_Nfin.update({new_param_key : new_finger_dict[new_param_key]})

        param_fingerConsider.update({math.sqrt(predict_energy**2+predict_energy_c1**2) : param_cosFin})
        param_Nfinger.update({math.sqrt(predict_energy**2+predict_energy_c1**2) : param_Nfin})
        const_dict.update({"{}".format(math.sqrt(predict_energy**2+predict_energy_c1**2)) : {'energy' : predict_energy, 'minCycle' : predict_minCycle, 'WM' : predict_WM, 'Taccess' : predict_Taccess, 'RM' : predict_RM,
                                                                                             'energy_c1' : predict_energy_c1, 'minCycle_c1' : predict_minCycle_c1, 'WM_c1' : predict_WM_c1, 'Taccess_c1' : predict_Taccess_c1, 'RM_c1' : predict_RM_c1}})
        # const_dict_1.update({"{}".format(predict_energy) : {'minCycle' : minCycleBase1, 'WM' : WMBase1, 'Taccess' : TaccessBase1, 'RM' : RMBase1}})
        # const_dict_2.update({"{}".format(predict_energy) : {'minCycle' : minCycleBase2, 'WM' : WMBase2, 'Taccess' : TaccessBase2, 'RM' : RMBase2}})
    # if(GPlearn and len(energyInConstraint_list)>1):    
    #     if (init1_energy_dr>max(drInConstraint_list[1:len(drInConstraint_list)])):
    #         opt=True
    # elif(GPlearn==False):
    #     if (math.sqrt(init1_predict_energy**2+init1_predict_energy_c1**2)>min(emInConstraint_list)):
    #         opt=True
    
    # if(opt==True):
    #     now_min_em=min(emInConstraint_list[1:len(emInConstraint_list)])
    #     now_min_const=const_dict[str(now_min_em)]
    #     now_min_paramF=param_fingerConsider[now_min_em]
    #     now_min_paramNF=param_Nfinger[now_min_em]
    
    with open("powerOPT.log_{}".format(today), "a") as LOG:
        midex=drInConstraint_list.index(max(drInConstraint_list))
        LOG.write('\n\n\n\n#### predict energy & constraint ####\n')
        LOG.write("iteration : "+str(k)+"\n")
        LOG.write("min energy : "+str(energyInConstraint_list[midex])+"\n")
        LOG.write("min energy_c1 : "+str(energyInConstraint_list_c1[midex])+"\n")
        LOG.write(str(current_point)+"\n")
        LOG.write(' energy : '+ str(predict_energy))
        LOG.write('\n minCycle : '+ str(predict_minCycle))
        LOG.write('\n Taccess : '+ str(predict_Taccess))
        LOG.write('\n RM : '+ str(predict_RM))
        LOG.write('\n WM : '+ str(predict_WM))
        LOG.write('\n energy_c1 : '+ str(predict_energy_c1))
        LOG.write('\n minCycle_c1 : '+ str(predict_minCycle_c1))
        LOG.write('\n Taccess_c1 : '+ str(predict_Taccess_c1))
        LOG.write('\n RM_c1 : '+ str(predict_RM_c1))
        LOG.write('\n WM_c1 : '+ str(predict_WM_c1))
        LOG.write('\n predict tcycle margin : '+ str(predict_tcycle_margin))
        LOG.write('\n predict taccess margin : '+ str(predict_Taccess_margin))
        LOG.write('\n predict tcycle margin_c1 : '+ str(predict_tcycle_margin_c1))
        LOG.write('\n predict taccess margin_c1 : '+ str(predict_Taccess_margin_c1))
        LOG.write('\n minCycle Const :'+ str(tcycleConst))
        LOG.write('\n Taccess Const :'+ str(TaccessConst))
        LOG.write('\n'+ str(energyInConstraint_list))
        LOG.write('\n'+ str(energyInConstraint_list_c1))
        LOG.write('\n'+ str(drInConstraint_list))
        # if(opt==True):
        #     LOG.write('\n now min energy : '+str(now_min_energy)+"\n")
        #     LOG.write('\n now min const : '+str(now_min_const)+"\n")
        #     LOG.write('\n now min finger Consider Param : '+str(now_min_paramF)+"\n")
        #     LOG.write('\n now min No finger Param : '+str(now_min_paramNF)+"\n")

        
    print(energyInConstraint_list)
    print(energyInConstraint_list_c1)
    print(drInConstraint_list)
    X_input.append(currentInputList)#
    y_tcycle.append(min(predict_tcycle_margin,predict_tcycle_margin_c1))#
    y_WM.append(max(predict_WM,predict_WM_c1))#
    y_taccess.append(min(predict_Taccess_margin,predict_Taccess_margin_c1))#
    y_RM.append(min(predict_RM,predict_RM_c1))#
    powerBO.register(params = current_point, target = loss_function)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TcycleGPR.fit(X_input, y_tcycle)
        WMGPR.fit(X_input, y_WM)
        TaccessGPR.fit(X_input, y_taccess)
        RMGPR.fit(X_input, y_RM)
        
    
    #next
    np.seterr(invalid="ignore")
    next_point = powerBO.suggest(utility, y_max = max(drInConstraint_list))
    # w_pg = next_point['SA_PG_width']
    # w_foot = next_point['SA_FOOT_width']
    if (pre_no==0):
        if (len(energyInConstraint_list[1:len(energyInConstraint_list)])==0):
            best_energy_list1.append(init1_predict_energy)
        else :
            best_energy_list1.append(min(energyInConstraint_list[1:len(energyInConstraint_list)]))
    else:
        mdx=drInConstraint_list.index(max(drInConstraint_list))
        best_energy_list1.append(energyInConstraint_list[mdx])
        best_energy_list1_c1.append(energyInConstraint_list_c1[mdx])
    best_energy_list2.append(powerBO.max["target"])
    
    powerGP_list.append(copy.deepcopy(powerBO._gp))
    TcycleGPR_list.append(copy.deepcopy(TcycleGPR))
    WMGPR_list.append(copy.deepcopy(WMGPR))
    TaccessGPR_list.append(copy.deepcopy(TaccessGPR))
    RMGPR_list.append(copy.deepcopy(RMGPR))
    
    # param_list.append(next_point)
    next_point_ary = np.array([np.fromiter(next_point.values(), dtype='float')])
    
    print("acq:", end="")
    print(utility.utility( next_point_ary, powerBO._gp, max(drInConstraint_list)))
    
    if(k >= bo_iter):
        break
#%%
opt_dr=drInConstraint_list.index(max(drInConstraint_list))
opt_index=DR_list.index(max(drInConstraint_list))

with open("powerOPT.log_{}".format(today), "a") as LOG:
    LOG.write("\n\ninit spec\n")
    LOG.write("energy : "+ str(init1_predict_energy )+ "\n")
    LOG.write("Tcycle : "+ str(init1_predict_minCycle )+ "\n")
    LOG.write("Taccess : "+ str(init1_predict_Taccess) + "\n")
    LOG.write("WM : "+ str(init1_predict_WM) + "\n")
    LOG.write("RM : "+ str(init1_predict_RM )+ "\n\n\n")
    LOG.write("\n ###### (optimize iter point / total interation count) : ({} / {}) ######".format(opt_index, bo_iter))
    LOG.write(" ##### opt result ####\n")
    LOG.write(" optimize Energy : {} \n".format(Energy_list[opt_index]))
    LOG.write(" optimize Tcycle : {} \n".format(minCycle_list[opt_index]))
    LOG.write(" optimize Taccess : {} \n".format(Taccess_list[opt_index]))
    LOG.write(" optimize WM : {} \n".format(WM_list[opt_index]))
    LOG.write(" optimize RM : {} \n".format(RM_list[opt_index]))
    LOG.write(" optimize Energy_c1 : {} \n".format(Energy_list_c1[opt_index]))
    LOG.write(" optimize Tcycle_c1 : {} \n".format(minCycle_list_c1[opt_index]))
    LOG.write(" optimize Taccess_c1 : {} \n".format(Taccess_list_c1[opt_index]))
    LOG.write(" optimize WM_c1 : {} \n".format(WM_list_c1[opt_index]))
    LOG.write(" optimize RM_c1 : {} \n".format(RM_list_c1[opt_index]))
    # LOG.write(" optimize constraint list : " + str(optConst))
    # LOG.write("\n optimize parameter : " + str(optParam))
    # LOG.write("\n optimize param consider finger : "+str(optParam_consFin))
    # LOG.write("\n opt finger number : "+ str(optfinger)+"\n")
    
if(LRmodel):
    optEnergy = min(energyInConstraint_list)
    optParam=param_dict["{}".format(optEnergy)]
    optConst = const_dict["{}".format(optEnergy)]
    mean_opt, sigma_opt = SAV.getPostSAos(optParam)
    nTRK_opt = round(optParam['Ntrk'])
    CHnTRK.TRKCH(instanceSize1, nTRK_opt)
    CHnTRK.TRKCH(instanceSize2, nTRK_opt)
    runSimulation(optParam,mean_opt,sigma_opt)
    
    result1_opt, result2_opt = extractSimResult(optParam, mean_opt, sigma_opt)
    energyBase1_opt = result1_opt[0]
    minCycleBase1_opt = result1_opt[1]
    WMBase1_opt = result1_opt[2]
    TaccessBase1_opt = result1_opt[3]
    RMBase1_opt= result1_opt[4]
    
    energyBase2_opt = result2_opt[0]
    minCycleBase2_opt = result2_opt[1]
    WMBase2_opt = result2_opt[2]
    TaccessBase2_opt = result2_opt[3]
    RMBase2_opt= result2_opt[4]

    predict_energy_opt = lp.linearPredict(col_base1, energyBase1_opt, col_base2, energyBase2_opt, col_target)
    predict_minCycle_opt = lp.linearPredict(col_base1, minCycleBase1_opt,col_base2, minCycleBase2_opt,col_target) #(minCycleBase1 + minCycleBase2)/2
    predict_Taccess_opt = lp.linearPredict(col_base1, TaccessBase1_opt, col_base2,TaccessBase2_opt , col_target)
    predict_RM_opt = lp.linearPredict(col_base1, RMBase1_opt, col_base2, RMBase2_opt, col_target)
    predict_WM_opt = (WMBase1_opt + WMBase2_opt)/2
    
    power_error_rate=(predict_energy_opt-optEnergy)/optEnergy*100
    tcycle_error_rate=(predict_minCycle_opt-optConst['minCycle'])/optConst['minCycle']*100
    taccess_error_rate=(predict_Taccess_opt-optConst['Taccess'])/optConst['Taccess']*100
    WM_error_rate=(predict_WM_opt-optConst['WM'])/optConst['WM']*100
    RM_error_rate=(predict_RM_opt-optConst['RM'])/optConst['RM']*100
    print("\npower error rate : {}".format(power_error_rate))
    print("\ntcycle error rate : {}".format(tcycle_error_rate))
    print("\ntaccess error rate : {}".format(taccess_error_rate))
    print("\nwm error rate : {}".format(WM_error_rate))
    print("\nrm error rate : {}".format(RM_error_rate))

best_energy_list1.insert(0, origin_energy)
print("time : ", time.time()-start)
with open("{}_par45_power.txt".format(instSize), 'w') as f:
    for i in best_energy_list1:
        line = "{}\n".format(i)
        f.write(line)
opt=True
if (opt==False and GPlearn==False):
    optEnergy = min(energyInConstraint_list)
    print("\n ###### No optimize ######")
    optParam = init_input1
    optConst = {'minCycle' : init1_predict_minCycle, 'WM' : init1_predict_WM, 'Taccess' : init1_predict_Taccess, 'RM' : init1_predict_RM}
    
    case=1
    
elif (opt==False and GPlearn==True):
    optEnergy = min_energy_old
    print("\n ##### old opt is better #####")
    optParam = min_param
    optConst = min_const
    case=2
    
else:
    optEnergy = min(energyInConstraint_list[1:len(energyInConstraint_list)])
    opt_point = Energy_list.index(optEnergy)+1
    print("\n ###### (optimize iter point / total interation count) : ({} / {}) ######".format(opt_point, bo_iter))
    optParam = param_dict["{}".format(optEnergy)]
    optEnergy1=param_dict_1["{}".format(optEnergy)]
    optEnergy2=param_dict_2["{}".format(optEnergy)]
    optConst = const_dict["{}".format(optEnergy)]
    optConst1 = const_dict_1["{}".format(optEnergy)]
    optConst2 = const_dict_2["{}".format(optEnergy)]
    case=3
optParam_consFin = param_fingerConsider[optEnergy]
optfinger = param_Nfinger[optEnergy]
opt_leakage1, opt_leakage2, opt_leakage=measLeakage(optParam)
with open("leakageOPT.log_power", "a") as LOG:
    LOG.write("opt leakage\n")
    LOG.write("opt leakage 1 : "+ str(opt_leakage1) + "\n")
    LOG.write("opt leakage 2 : "+ str(opt_leakage2) + "\n")
    LOG.write("opt predict leakage : "+ str(opt_leakage) + "\n\n\n")

    
            
    
    
print(" ##### opt result ####\n")
print(" optimize Energy : {} \n".format(optEnergy))
print(" optimize constraint list : ", optConst)
print("\n optimize parameter : ", optParam)
print("\n opt finger consider parameter : ", optParam_consFin)
print("\n opt finger number : "+ str(optfinger))


with open("powerOPT.log_{}".format(today), "a") as LOG:
    LOG.write("\n\ninit spec\n")
    LOG.write("energy : "+ str(init1_predict_energy )+ "\n")
    LOG.write("Tcycle : "+ str(init1_predict_minCycle )+ "\n")
    LOG.write("Taccess : "+ str(init1_predict_Taccess) + "\n")
    LOG.write("WM : "+ str(init1_predict_WM) + "\n")
    LOG.write("RM : "+ str(init1_predict_RM )+ "\n\n\n")
    if(case==1):
        LOG.write("\n no optimize, initial value is minimum energy")
        LOG.write(" ##### opt result ####\n")
        LOG.write(" optimize Energy : {} \n".format(optEnergy))
        LOG.write(" optimize constraint list : " + str(optConst))
        LOG.write("\n optimize parameter : " + str(optParam))
    elif(case==2):
        LOG.write("\n old OPT is better than new OPT\n")
        LOG.write(" optimize Energy : {} \n".format(min_energy_old))
        LOG.write(" optimize Taccess : {} \n".format(min_taccess_old))
        LOG.write(" optimize Tcycle : {} \n".format(min_tcycle_old))
        LOG.write(" optimize WM : {} \n".format(min_wm_old))
        LOG.write(" optimize RM : {} \n".format(min_rm_old))
        LOG.write("\n optimize parameter : " + str(min_param))
    else:
        LOG.write("\n ###### (optimize iter point / total interation count) : ({} / {}) ######".format(opt_point, bo_iter))
        LOG.write(" ##### opt result ####\n")
        LOG.write(" optimize Energy : {} \n".format(optEnergy))
        LOG.write(" optimize constraint list : " + str(optConst))
        LOG.write("\n optimize parameter : " + str(optParam))
        LOG.write("\n optimize param consider finger : "+str(optParam_consFin))
        LOG.write("\n opt finger number : "+ str(optfinger)+"\n")
        if (GPlearn):
            LOG.write("\n pre iter : "+ str(pre_iter_num)+"\n")
            if (min_energy_old<min(energyInConstraint_list)):
                LOG.write("\n old OPT is better than new OPT\n")
                LOG.write(" optimize Energy : {} \n".format(min_energy_old))
                LOG.write(" optimize Taccess : {} \n".format(min_taccess_old))
                LOG.write(" optimize Tcycle : {} \n".format(min_tcycle_old))
                LOG.write(" optimize WM : {} \n".format(min_wm_old))
                LOG.write(" optimize RM : {} \n".format(min_rm_old))
                LOG.write("\n optimize parameter : " + str(min_param))
    

opt_leakage1, opt_leakage2, opt_leakage=measLeakage(optParam)
with open("leakageOPT.log_power", "a") as LOG:
    LOG.write("opt leakage\n")
    LOG.write("opt leakage 1 : "+ str(opt_leakage1) + "\n")
    LOG.write("opt leakage 2 : "+ str(opt_leakage2) + "\n")
    LOG.write("opt predict leakage : "+ str(opt_leakage) + "\n\n\n")

optMinCycle = optConst['minCycle']
optWM = optConst['WM']
optTaccess = optConst['Taccess']
optRM = optConst['RM']

optParamResult = []
for i in paramList:
    optParamResult.append(optParam[i])

if (min(energyInConstraint_list)==init1_predict_energy):
    optSp = lisFile+'_opt.sp'
    optSp_Vth0 = lisFile_Vth0+'_opt.sp'
    mean = init1_mean
    sigma = init1_sigma
else:
    optSp = lisFile+'_opt.sp'
    optSp_Vth0 = lisFile_Vth0+'_opt.sp'
    optPoint = Energy_list.index(optEnergy)
    mean = mean_list[optPoint]
    sigma = sigma_list[optPoint]


# [w_pg_opt, w_foot_opt] = [optParam['SA_PG_width'], optParam['SA_FOOT_width']]
# subprocess.call("./getSAvos.py {} 30e-9 2000e-9 90e-9 500e-9 30e-9 {} 30e-9 1".format(w_pg_opt, w_foot_opt), shell=True)

# f = open('./Vos.txt', 'r')
# lines = f.readlines()
# b = lines[0].split()
# mean = -float(b[1][:-1])
# sigma = -float(b[3][:-2])
# f.close()
# mean = 0.000144799051832427 #-float(b[1][:-1])
# sigma = 0.00491978824373201 #-float(b[3][:-2])

print("\norigin energy : "+str(origin_energy)+"\n\n")
pv1 = re.compile('v1 ')
a = [pv1]
with open(optSp,'w') as fw:
    with open(ref,'r') as f:
        fline = f.readlines()
        for line in fline:
            for i, j, k in zip(optParamResult, range(1, len(optParamResult)+1), a):
                if(k.search(line)):
                    line = line.replace("v{}".format(j) , "%s" )%float(i)                                 
            fw.write(line)

with open(optSp_Vth0, 'w') as fw:
    with open(ref_Vth0, 'r') as f:
        fline = f.readlines()
        for line in fline:
            for i, j, k in zip(optParamResult, range(1, len(optParamResult)+1), a):
                if(k.search(line)):
                    line = line.replace("v{}".format(j) , "%s" )%float(i)                                 
            if(VosMean.search(line)):
                line = line.replace("VosCH1" , "%s" )%float(mean)                    
            if(VosSigma.search(line)):
                line = line.replace("VosCH2" , "%s" )%float(sigma)                    
            fw.write(line)

# nTRK_opt = round(optParam['Ntrk'])
# subprocess.call("./TRKBcell_onNumChange.py {} {}".format(instSize, nTRK_opt), shell=True)
# subprocess.call("cp {0}_TRKch.ckt {0}_TRKch_opt.ckt".format(instSize), shell=True)
# subprocess.call("cp {0}_dvth_TRKch.ckt {0}_dvth_TRKch_opt.ckt".format(instSize), shell=True)



##### change the input netlist in cycle_PCH_opt.sp before simulation ####


plt.figure(1)
plt.ylabel('Tcycle')
plt.xlabel('Energy')
plt.title('Tcycle Constraint_0.9v', fontdict = {'weight':'bold', 'size':18})
# plt.plot(initEnergy, initCycle[0], 'bo')
plt.plot(Energy_list, y_tcycle[2:], 'ro')
#plt.plot(optEnergy, optMinCycle, 'b+')
plt.axhline(tcycleConst, 0, 1, color='gray', linestyle='--', linewidth=3)
plt.axis([0,max(Energy_list)+2e-12,0,tcycleConst + 1e-10])
plt.show()

plt.figure(2)
plt.ylabel('Taccess')
plt.xlabel('Energy')
plt.title('Taccess Constraint_0.9v', fontdict = {'weight':'bold', 'size':18})
plt.plot(init2_predict_energy, init2_predict_Taccess, 'bo')
plt.plot(Energy_list, y_taccess[2:], 'ro')
plt.plot(optEnergy, optTaccess, 'b+')
plt.axhline(TaccessConst, 0, 1, color='gray', linestyle='--', linewidth=3)
plt.axis([0,max(Energy_list)+2e-12, 0, TaccessConst+1e-10])
plt.show()

plt.figure(3)
plt.ylabel('WM')
plt.xlabel('Energy')
plt.title('Write Margin Constraint_0.9v', fontdict = {'weight':'bold', 'size':18})
# plt.plot(initEnergy, initWM[0], 'bo')
plt.plot(Energy_list, y_WM[2:], 'ro')
plt.plot(optEnergy, optWM, 'b+')
plt.axhline(WMConst, 0, 1, color='gray', linestyle='--', linewidth=3)
plt.axis([0,max(Energy_list)+2e-12, 0, 1])
plt.show()

plt.figure(4)
plt.ylabel('RM')
plt.xlabel('Energy')
plt.title('Read Margin Constraint_0.9v', fontdict = {'weight':'bold', 'size':18})
# plt.plot(initEnergy, initRM[0], 'bo')
plt.plot(Energy_list, y_RM[2:], 'ro')
plt.plot(optEnergy, optRM, 'b+')
plt.axhline(RMConst, 0, 1, color='gray', linestyle='--', linewidth=3)
plt.axis([0,max(Energy_list)+2e-12, 0, 1])
plt.show()

plt.figure(5)
plt.ylabel('Energy')
plt.xlabel('iteration')
plt.title('iteration_0.9v', fontdict = {'weight':'bold', 'size':18})
plt.plot(np.arange(0, len(best_energy_list1), 1), best_energy_list1, 'r--')

# plt.axhline(tcycleConst, 0, 1, color='gray', linestyle='--', linewidth=3)
plt.axis([0, bo_iter, min(best_energy_list1)-min(best_energy_list1)/100*5, max(best_energy_list1)+max(best_energy_list1)/100*5])
plt.show()

plt.figure(6)
plt.ylabel('Energy')
plt.xlabel('iteration')
plt.title('iteration_0.9v', fontdict = {'weight':'bold', 'size':18})
plt.plot(np.arange(0, len(energyInConstraint_list), 1), energyInConstraint_list, 'r--')

# plt.axhline(tcycleConst, 0, 1, color='gray', linestyle='--', linewidth=3)
plt.axis([0, bo_iter, 0, max(best_energy_list1)])
plt.show()

#%%

# optEnergy = min(energyInConstraint_list)
# optParam = param_dict["{}".format(optEnergy)]

# ######### powerBO.max run ######
# b1 = powerBO.max['params'][param1] 
# b2 = powerBO.max['params'][param2] 

# #b7 = current_point["AGIDL"]
# final_loss, final_minCycle = inserthspiceinput(b1, b2)

# OperationPower=[]
# with open ("write_power.txt", "r") as f:
#         fline = f.readlines()
#         for line in fline:
#             line=float(line)
#             OperationPower.append(float(line))
            
# init_loss, init_minCycle = inserthspiceinput(initial_point11,initial_point12)
          
# loss_function=final_loss
# print("init loss is ", init_loss)
# print("min loss is ", loss_function)
# print("mincycle time :", final_minCycle)
# print(loss_function)
os.chdir(path)
if (GPlearn==False):
    os.system("cp powerOPT.log_{}_power powerOPT.log_{}_origin".format(today, today))
rmf.rfile('fingerChanging')
# os.system("rm -rf *offset1_*")
# os.system("rm -rf *offset2_*")
    
print("time : ", time.time()-start)

