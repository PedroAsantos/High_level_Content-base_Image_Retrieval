import urllib.request
import urllib.error
import re
import socket
socket.setdefaulttimeout(5)
file = open("imagenet_fall11_urls/fall11_urls.txt", "r")
i=0
c=0
cc=0
MAX_IMAGES=100
posFile = range(0,100000,10000)
for line in file:
    if i>posFile[c]:
        lst = re.split('\s+', line)
        try:
            print(lst[1])
            urllib.request.urlretrieve(lst[1], "image_"+str(i)+".jpg")
        except urllib.error.HTTPError as e:
            print(e)
            i-=1
        except urllib.error.URLError as e:
            print(e)
            i-=1
        except socket.timeout as e:
            print(e)
            i-=1
        print(i)
        cc+=1
        if cc==10:
            c+=1
            cc=0
    i+=1
    #if i==MAX_IMAGES:
    #    break
