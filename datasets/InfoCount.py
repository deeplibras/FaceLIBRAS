import os

basePath = 'CK+'

infoFile = os.path.join(os.path.dirname(__file__), basePath+'/info.txt')

trainLine = None
count = [0,0,0,0,0,0,0]
with open(infoFile) as f:
	for line in f:
		info = line.split('@')

		count[int(info[1])] = count[int(info[1])] + 1

print(count)

'''
KDEF
0 = 418 = Afraid
1 = 420 = Angry
2 = 418 = Disgusted
3 = 420 = Happy
4 = 418 = Neutral
5 = 419 = Sad
6 = 417 = Surprised


CK+
0 = 44 = Anger
1 = 17 = Contempt
2 = 58 = Disgust
3 = 25 = Fear
4 = 67 = Happy
5 = 28 = Sadness
6 = 81 = Surprise
'''
