from PIL import Image
import PIL.ImageOps   
from PIL import ImageFilter
import re
import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
#img = Image.open("screenshot_p4.png");
img = cv2.imread('screenshot_p8.png')
img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
img=cv2.filter2D(img,-1,mask)
threshold = 180 # to be determined
# _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)


# img = Image.fromarray(img_binarized)



img = Image.fromarray(img)
img.show()
#img = img.filter(ImageFilter.SHARPEN)

text = pytesseract.image_to_string(img, lang='eng')
print(text)
lineup = {'DUKE':[], 'ARCH':[], 'SCIE':[]} #
li = list(filter(None, re.split(' |\n|]|\)', text)))
for i in range(len(li)):
    if '#2006' in li[i]:
        if ('DUKE' in li[i-1]) or ('ARCH' in li[i-1]) or ('SCIE' in li[i-1]):
            for j in [li[i+1], li[i+2], li[i+3], [i+4]]:
                if 'X:' in j:
                    x_coord = j[2:]
                if 'Y:' in j:
                    y_coord = j[2:]
            lineup[li[i-1]].append((x_coord, y_coord))
print(li)
print(lineup)

# import pyautogui
# import cv2
# import numpy as np


# chat_lt = (479, 180)
# chat_rb =(1141, 959)
# img = pyautogui.screenshot(region=[chat_lt[0],chat_lt[1],chat_rb[0]-chat_lt[0],chat_rb[1]-chat_lt[1]]) # x,y,w,h

# img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
# # cv2.imshow('img',img)
# # cv2.waitKey(0)
# cv2.imwrite('screenshot.png',img)