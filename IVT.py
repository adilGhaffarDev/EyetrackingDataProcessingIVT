import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
#here are some additional settings that may help: There was a 3 x 3 points calibration matrix used, upper left point had coordinates -700, -700, and bottom right 700, 700, center point was at 0,0.  (These were not pixels, let's call them units)

#he physical size of the bounding rectangle for the 3x3 matrix was 195 mm (horizontally) and 113mm vertically. It means that one degree was approximately 97 units horizontally, and 56 vertically.
# fixation veocity threshold <100deg/sec
# saccade velocity thresold >300deg/sec
#Subjects were seated at 450mm from the screen.
#Group 5 - s5, s15, s25, s1, s11, s21
frequency = 1000
velocityThreshold = 100 #deg/sec
timeThresoldFixation = 0.08 #sec
ourUsers = ['s5','s15','s25','s1','s11','s21']
s5_True = []
s5_False = []
s15_True = []
s15_False = []
s25_True = []
s25_False = []
s1_True = []
s1_False = []
s11_True = []
s11_False = []
s21_True = []
s21_False = []

#Calculates mean fixation durations(MFA) and mean saccade amplitudes(MSA) and their standard deviations(SD)
def MFD_MSA(fixationSaccadeData):
    finalresult = []
    MSAs = []
    MFDs = []
    for data in fixationSaccadeData:
        #centroids = []
        SAs = data[0]
        FCs = [p[1] for p in data[1]]
        MSAs.append(np.mean(SAs))
        MFDs.append(np.mean(FCs))
    
    MSA = np.mean(MSAs)
    MFD = np.mean(MFDs)
    SD_MSA = np.std(MSAs)
    SD_MFD = np.std(MFDs)
    finalresult.append(MSA)
    finalresult.append(SD_MSA)
    finalresult.append(MFD)
    finalresult.append(SD_MFD)

    return finalresult

#Calculates overall MSAs and MFDs
def GetOverallMSD_MFA(MSA_MFD_True,MSA_MFD_False):
    s_Overall_MSA = np.mean([MSA_MFD_True[0],MSA_MFD_False[0]])
    s_Overall_MSA_SD = np.mean([MSA_MFD_True[1],MSA_MFD_False[1]])
    s_Overall_MFD = np.mean([MSA_MFD_True[2],MSA_MFD_False[2]])
    s_Overall_MFD_SD = np.mean([MSA_MFD_True[3],MSA_MFD_False[3]])
    result = []
    result.append(s_Overall_MSA)
    result.append(s_Overall_MSA_SD)
    result.append(s_Overall_MFD)
    result.append(s_Overall_MFD_SD)
    return result

#plotting of individual graphs against user ids
def PlotBarGraph(true_means,false_means,overall_means,true_std,false_std,overall_std,yLabel,title):
    ind = np.arange(len(true_means))  # the x locations for the groups
    width = 0.9  # the width of the bars

    fig,ax = plt.subplots()
    ax.bar(ind - width/3, true_means, width/3, yerr=true_std,
                label='True', capsize=4)
    ax.bar(ind, false_means, width/3, yerr=false_std,
                label='False', capsize=4)
    ax.bar(ind + width/3, overall_means, width/3, yerr=overall_std,
                label='Overall', capsize=4)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(yLabel)
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(ourUsers)
    ax.legend()

    fig.tight_layout()

    plt.show()

#plotting of aggregated graphs against trues and falses
def PlotAgregatedGraph(true_means,false_means,true_std,false_std,yLabel,title):
    ind = np.arange(2)  # the x locations for the groups
    allTruesMean = np.mean(true_means)
    allFalseMean = np.mean(false_means)
    allTruesStd = np.std(true_means)
    allFalseStd = np.std(false_means)
    fig,ax = plt.subplots()
    ax.bar(ind, [allTruesMean,allFalseMean],yerr=[allTruesStd,allFalseStd], align='center', alpha=0.5, ecolor='black', capsize=10)
    

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(yLabel)
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(['True','False'])
    ax.legend()

    fig.tight_layout()

    plt.show()

#this function finds centriods for the fixations and also finds saccade amplitudes
def FinalResultIVT(fixationData1):
    finalresult = []
    centroids = []
    centroidsTime = []
    
    #centroids of fixations and time
    for data in fixationData1:
        #centroids = []
        x = [p[0] for p in data]
        y = [p[1] for p in data]
        centroid = (sum(x) / len(data), sum(y) / len(data))
        time = len(x)/frequency
        #centroids.append([centroid,time])
        if time>=timeThresoldFixation:
            centroidsTime.append([centroid,time])
            centroids.append(centroid)
    
    #saccade amplitudes
    saccadeApmlitudes = []
    if len(centroids) > 0:
        prevPointDeg = (0,0) #conversion of units to degree
        t = 0;
        for center in centroids:
            currPointDeg = (float(center[0])/97,float(center[1])/56)#conversion of units to degree
            if t != 0:
                dstDeg = distance.euclidean(prevPointDeg, currPointDeg)
                saccadeApmlitudes.append(dstDeg)
            prevPointDeg = currPointDeg
            t = t+1
    
    finalresult.append(saccadeApmlitudes)
    finalresult.append(centroidsTime)
    return finalresult

#draws scatter plot for fixations for testing
def ScatterPlot(fixpoints,saccadepoints,centers):
    plt.scatter([p[0] for p in saccadepoints],[p[1] for p in saccadepoints],color='black',alpha=0.5)
    plt.scatter([p[0] for p in fixpoints],[p[1] for p in fixpoints],color='red',alpha=0.5)
    RADIUS = 10
    
    #     pyplot.plot(data[:,[0]], data[:,[1]])
    pp = [p[0] for p in centers[1]]
    for point in pp:
        plt.scatter(point[0],point[1],color='yellow')
        center = plt.Circle((point[0],point[1]), RADIUS, fill = False, color ='blue' )
        plt.gcf().gca().add_artist(center)
    plt.title("Fixtation detected")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()

#IVT algo to detect fixations
def IVTFixationPro(userdata):

    time = 1/frequency
    simplefixationsforscatter = []
    currentSampleFixation = []
    currentSampleSaccade = []
    fix = []
    #print(a[0])
    prevPointDeg = (float(userdata[0])/97,float(userdata[1])/56) #conversion of units to degree
    prevPointUnit = (float(userdata[0]),float(userdata[1])) 
    for i in range(2,len(userdata),2):
        currPointDeg = (float(userdata[i])/97,float(userdata[i+1])/56)#conversion of units to degree
        currPointUnit = (float(userdata[i]),float(userdata[i+1]))#conversion of units to degree
        dstDeg = distance.euclidean(prevPointDeg, currPointDeg)
        dstUnit = distance.euclidean(prevPointUnit, currPointUnit)
        velocity = dstDeg/time
        velocityU = dstUnit/time

        if velocity <= velocityThreshold:
            fix.append(prevPointUnit)
            simplefixationsforscatter.append(prevPointUnit)
        else:
            if len(fix)>0:
                fix.append(currPointUnit)
                simplefixationsforscatter.append(currPointUnit)
                fixTemp = list(fix)
                currentSampleFixation.append(fixTemp)
            currentSampleSaccade.append(prevPointUnit)
            fix.clear()
            
        prevPointDeg = currPointDeg
        prevPointUnit = currPointUnit
        
    if len(fix)>0:
        fixTemp = list(fix)
        currentSampleFixation.append(fixTemp)
#    ScatterPlot(simplefixationsforscatter,currentSampleSaccade,FinalResultIVT(currentSampleFixation))
    return FinalResultIVT(currentSampleFixation)

#main call
#Reading train.csv
with open('train.csv') as csv_file:
    rows = csv.reader(csv_file, delimiter=',')
    
    for row in rows:
        if row[0] in ourUsers:
            if row[0] == ourUsers[0] and row[1] == 'true':
                    s5_True.append(row[2:])
            if row[0] == ourUsers[0] and row[1] == 'false':
                    s5_False.append(row[2:])
            
            if row[0] == ourUsers[1] and row[1] == 'true':
                    s15_True.append(row[2:])
            if row[0] == ourUsers[1] and row[1] == 'false':
                    s15_False.append(row[2:])
            
            if row[0] == ourUsers[2] and row[1] == 'true':
                    s25_True.append(row[2:])
            if row[0] == ourUsers[2] and row[1] == 'false':
                    s25_False.append(row[2:])
            
            if row[0] == ourUsers[3] and row[1] == 'true':
                    s1_True.append(row[2:])
            if row[0] == ourUsers[3] and row[1] == 'false':
                    s1_False.append(row[2:])
                    
            if row[0] == ourUsers[4] and row[1] == 'true':
                    s11_True.append(row[2:])
            if row[0] == ourUsers[4] and row[1] == 'false':
                    s11_False.append(row[2:])
                    
            if row[0] == ourUsers[5] and row[1] == 'true':
                    s21_True.append(row[2:])
            if row[0] == ourUsers[5] and row[1] == 'false':
                    s21_False.append(row[2:])
    
    s5_True_IVTResult = []
    s5_False_IVTResult = []
    
    s15_True_IVTResult = []
    s15_False_IVTResult = []
    
    s15_True_IVTResult = []
    s15_False_IVTResult = []
    
    s25_True_IVTResult = []
    s25_False_IVTResult = []
    
    s1_True_IVTResult = []
    s1_False_IVTResult = []
    
    s11_True_IVTResult = []
    s11_False_IVTResult = []
    
    s21_True_IVTResult = []
    s21_False_IVTResult = []
    
    print('1')
    for data in s5_True:
        s5_True_IVTResult.append(IVTFixationPro(data)) 
    
    print('2')
    for data in s5_False:
        s5_False_IVTResult.append(IVTFixationPro(data))
    
    for data in s15_True:
        s15_True_IVTResult.append(IVTFixationPro(data)) 
    print('4')
    for data in s15_False:
        s15_False_IVTResult.append(IVTFixationPro(data))
   
    print('5')
    for data in s25_True:
        s25_True_IVTResult.append(IVTFixationPro(data))
    print('6')
    for data in s25_False:
        s25_False_IVTResult.append(IVTFixationPro(data))
    
    print('7')
    for data in s1_True:
        s1_True_IVTResult.append(IVTFixationPro(data))
    print('8')
    for data in s1_False:
        s1_False_IVTResult.append(IVTFixationPro(data))
    
    print('9')
    for data in s11_True:
        s11_True_IVTResult.append(IVTFixationPro(data))
    print('10')
    for data in s11_False:
        s11_False_IVTResult.append(IVTFixationPro(data))
    
    print('11')
    for data in s21_True:
        s21_True_IVTResult.append(IVTFixationPro(data))
    print('12')
    for data in s21_False:
        s21_False_IVTResult.append(IVTFixationPro(data))
     
#    part 3 (Calculation of mean dixation durations(MFA) and mean saccade amplitudes(MSA))
    s5_True_MFD_MSA = MFD_MSA(s5_True_IVTResult)
    s5_False_MFD_MSA = MFD_MSA(s5_False_IVTResult)
    s5_Overall_MFD_MSA = GetOverallMSD_MFA(s5_True_MFD_MSA,s5_False_MFD_MSA)
    
    s15_True_MFD_MSA = MFD_MSA(s15_True_IVTResult)
    s15_False_MFD_MSA = MFD_MSA(s15_False_IVTResult)
    s15_Overall_MFD_MSA = GetOverallMSD_MFA(s15_True_MFD_MSA,s15_False_MFD_MSA)

    
    s25_True_MFD_MSA = MFD_MSA(s25_True_IVTResult)
    s25_False_MFD_MSA = MFD_MSA(s25_False_IVTResult)
    s25_Overall_MFD_MSA = GetOverallMSD_MFA(s25_True_MFD_MSA,s25_False_MFD_MSA)

    
    s1_True_MFD_MSA = MFD_MSA(s1_True_IVTResult)
    s1_False_MFD_MSA = MFD_MSA(s1_False_IVTResult)
    s1_Overall_MFD_MSA = GetOverallMSD_MFA(s1_True_MFD_MSA,s1_False_MFD_MSA)

    
    s11_True_MFD_MSA = MFD_MSA(s11_True_IVTResult)
    s11_False_MFD_MSA = MFD_MSA(s11_False_IVTResult)
    s11_Overall_MFD_MSA = GetOverallMSD_MFA(s11_True_MFD_MSA,s11_False_MFD_MSA)

    
    s21_True_MFD_MSA = MFD_MSA(s21_True_IVTResult)
    s21_False_MFD_MSA = MFD_MSA(s21_False_IVTResult)
    s21_Overall_MFD_MSA = GetOverallMSD_MFA(s21_True_MFD_MSA,s21_False_MFD_MSA)
    
    True_MFD_MSAs = [s5_True_MFD_MSA, s15_True_MFD_MSA, s25_True_MFD_MSA, s1_True_MFD_MSA, s11_True_MFD_MSA, s21_True_MFD_MSA]
    False_MFD_MSAs = [s5_False_MFD_MSA, s15_False_MFD_MSA, s25_False_MFD_MSA, s1_False_MFD_MSA, s11_False_MFD_MSA, s21_False_MFD_MSA]
    Overall_MFD_MSAs = [s5_Overall_MFD_MSA, s15_Overall_MFD_MSA, s25_Overall_MFD_MSA, s1_Overall_MFD_MSA, s11_Overall_MFD_MSA, s21_Overall_MFD_MSA]
   
    #subject_id   MFD_true MFD_SD_true MFD_false MFD_SD_false MSA_true MSA_SD_true MSA_false MSA_SD_false MFD_overall MFD_overall_SD MSA_overall MSA_overall_SD
    #csv genration for part 3
    exists = os.path.isfile('Part3.csv')
    if exists:
        os.remove('Part3.csv')
    with open('Part3.csv', 'x') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow(['subject_id', 'MFD_true', 'MFD_SD_true', 'MFD_false', 'MFD_SD_false', 'MSA_true', 'MSA_SD_true', 'MSA_false', 'MSA_SD_false', 'MFD_overall', 'MFD_overall_SD', 'MSA_overall' ,'MSA_overall_SD'])
        for i in range(len(ourUsers)):
            rowData = [str(ourUsers[i]), str(True_MFD_MSAs[i][2]), str(True_MFD_MSAs[i][3]), str(False_MFD_MSAs[i][2]),str(False_MFD_MSAs[i][3]), str(True_MFD_MSAs[i][0]), str(True_MFD_MSAs[i][1]), str(False_MFD_MSAs[i][0]), str(False_MFD_MSAs[i][1]), str(Overall_MFD_MSAs[i][2]), str(Overall_MFD_MSAs[i][3]), str(Overall_MFD_MSAs[i][0]) ,str(Overall_MFD_MSAs[i][1])]
            filewriter.writerow(rowData)

    print("Done");
    csvfile.close()

    print('done')
    
    #part4
    #Plotting graph for analysis
    PlotBarGraph([p[2] for p in True_MFD_MSAs],[p[2] for p in False_MFD_MSAs],[p[2] for p in Overall_MFD_MSAs], [p[3] for p in True_MFD_MSAs],[p[3] for p in False_MFD_MSAs],[p[3] for p in Overall_MFD_MSAs],'MFD','Mean Fixation Duration Graph')
    PlotBarGraph([p[0] for p in True_MFD_MSAs],[p[0] for p in False_MFD_MSAs],[p[0] for p in Overall_MFD_MSAs], [p[1] for p in True_MFD_MSAs],[p[1] for p in False_MFD_MSAs],[p[1] for p in Overall_MFD_MSAs],'MSA','Mean Saccade Amplitude Graph')
    PlotAgregatedGraph([p[2] for p in True_MFD_MSAs],[p[2] for p in False_MFD_MSAs], [p[3] for p in True_MFD_MSAs],[p[3] for p in False_MFD_MSAs],'MFD','Mean Fixation Duration Graph Aggregated')
    PlotAgregatedGraph([p[0] for p in True_MFD_MSAs],[p[0] for p in False_MFD_MSAs], [p[1] for p in True_MFD_MSAs],[p[1] for p in False_MFD_MSAs],'MSA','Mean Saccade Amplitude Graph Aggregated')
