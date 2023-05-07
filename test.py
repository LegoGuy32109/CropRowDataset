# 240 tall x 320 wide
import tensorflow as tf
import numpy as np
from PIL import Image
import io
# tesId + 1 => file number
testIds = []
# for i in range(281):
#     num = "%03d" % (i+1,)
#     print(num)
#     try:
#         arr = np.load('train_labels/train_labels/crop_row_'+str(num)+'.npy')
#         img = tf.keras.utils.array_to_img(arr)
#         img.save("masks/mask"+str(num)+".jpg")
#     except:
#         print("no ",num)
#         testIds.append(i)

# print(testIds)
counter = 0
for i in range(281):
    num = "%03d" % (i+1,)
    try:
        arr = np.load('train_labels/train_labels/crop_row_'+str(num)+'.npy')
        try:
            img = Image.open('Images/Images/crop_row_'+str(num)+'.jpg')
            img.save('training_data\crop_row_'+str(counter)+'.jpg')
            np.save('training_data\crop_row_'+str(counter), arr)
            counter+=1
        except:
            print("Couldn't load")
    except:
        print("couldn't find label for", num)