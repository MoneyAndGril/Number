import numpy as np
from keras.models import load_model    # 模型加载
import cv2
# import mnist_test
# cpu 运行
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = load_model('./weights.32-0.9965.hdf5')
# model.summary()
img = cv2.imread('./pic\\mnist_pic\\11.jpg')
# img = cv2.imread('D:\\python_game\\game_hyol\\MyMinst\\pic\\HandDraw\\18.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(gray.shape)
cv2.imshow('img',gray)
cv2.waitKey(0)
# scale = pix / height
# height = pix
change_img = cv2.resize(gray,(28,28))
# cv2.imshow('img',change_img)
# cv2.waitKey(0)
# cv2.imwrite('D:\\python_game\\game_hyol\\MyMinst\\pic\\mnist_pic\\18.jpg',change_img)
# cv2.imwrite('D:\\python_game\\game_hyol\\MyMinst\\pic\\45.png',change_img)
# cv2.destroyAllWindows()

# print(type(change_img))

# img_np = np.resize(change_img,[1,28,28,1])
img_np = change_img.reshape([-1,28,28,1]) / 255.0
print(img_np.shape)
#
a = model.predict(img_np,batch_size=1)

print(np.argmax(a,axis=1))
# print(a)

# img = cv2.imread('D:\\python_game\\game_hyol\\MyMinst\\pic\\TelCamera\\22.jpg')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(img.shape)
# img = (255-img)/255.0
# cv2.imshow('img',img)
# cv2.waitKey(0)
# img = cv2.resize(img,(28,28))
# retval, img = cv2.threshold(i
# mg, 127, 255, cv2.THRESH_BINARY_INV)
# img = 1-img
# img = (255-img)/255.0
# img = cv2.resize(img,(28,28))
# print(img)
# print(img.shape)
# img = cv2.resize(img,(28,28))

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.imwrite('D:\\python_game\\game_hyol\\MyMinst\\pic\\Handle_TelCamera\\22_28.png',img)
# cv2.destroyAllWindows()

path = './pic\\HandDraw'
import os
for dir in os.listdir(path):
    # child_dir : A B C
    print(dir)
    child_dir = os.path.join(path, dir)
    img = cv2.imread(child_dir)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_pic = cv2.resize(img,(28,28))
    img = np.reshape(img_pic,[1,28,28,1]) / 255.0
    # print(img.shape)
    # img = img / 255.0
    y = model.predict(img,batch_size=1)
    y = np.argmax(y,axis=1)
    s = dir.split('.')
    y = s[0]+'_'+str(y)+'.png'
    print(y)

    cv2.imwrite(f'./pic\\write\\{y}',img_pic)