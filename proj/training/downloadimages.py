import csv
import subprocess
import os

openImageDir = '/data-hdd/sunita/openImages/'

runMode = "train"

classes = ["Tree"]

with open('../class-descriptions-boxable.csv', mode='r') as infile:
    reader = csv.reader(infile)
    dict_list = {rows[1]:rows[0] for rows in reader}

subprocess.run(['rm', '-rf', 'JPEGImagesTrees'])
subprocess.run([ 'mkdir', 'JPEGImagesTrees'])

subprocess.run(['rm', '-rf', 'labelsTrees'])
subprocess.run([ 'mkdir', 'labelsTrees'])

for ind in range(0, len(classes)):

    className = classes[ind]
    print(str(ind) + " : " + className)

    commandStr = "grep "+dict_list[className] + " ../" + runMode + "-annotations-bbox.csv"
    class_annotations = subprocess.run(commandStr.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
    class_annotations = class_annotations.splitlines()

    totalNumOfAnnotations = len(class_annotations)
    print("Total number of annotations : "+str(totalNumOfAnnotations))

    cnt = 0
    for line in class_annotations[0:totalNumOfAnnotations]:
        cnt = cnt + 1
        print("annotation count : " + str(cnt))
        lineParts = line.split(',')
        #if (float(lineParts[8])>0 or float(lineParts[9])>0 or float(lineParts[10])>0 or float(lineParts[11])>0 or float(lineParts[12])>0):
        #    print("Skipped %s",lineParts[0])
        #    continue
        subprocess.run([ 'aws', 's3', '--no-sign-request', '--only-show-errors', 'cp', 's3://open-images-dataset/'+runMode+'/'+lineParts[0]+".jpg", 'JPEGImagesTrees/'+lineParts[0]+".jpg"])
        with open('labelsTrees/%s.txt'%(lineParts[0]),'a') as f:
            f.write(' '.join([str(ind),str((float(lineParts[5]) + float(lineParts[4]))/2), str((float(lineParts[7]) + float(lineParts[6]))/2), str(float(lineParts[5])-float(lineParts[4])),str(float(lineParts[7])-float(lineParts[6]))])+'\n')
