import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

num_images = 210

image_names = glob.glob("train_images/*.jpg")
image_names.sort()
image_names_subset = image_names[0:num_images]
images = [cv2.imread(img, 0) for img in image_names_subset]
image_dataset = np.array(images)
image_dataset = np.expand_dims(image_dataset, axis = 3)

mask_names = glob.glob("train_masks/*.jpg")
mask_names.sort()
mask_names_subset = mask_names[0:num_images]
masks = [cv2.imread(mask, 0) for mask in mask_names_subset]
mask_dataset = np.array(masks)
mask_dataset = np.expand_dims(mask_dataset, axis = 3)

print("Image data shape is: ", image_dataset.shape)
print("Mask data shape is: ", mask_dataset.shape)
print("Max pixel value in image is: ", image_dataset.max())
print("Labels in the mask are : ", np.unique(mask_dataset))

#Normalize images
image_dataset = image_dataset /255.  #Can also normalize or scale using MinMax scaler
#Do not normalize masks, just rescale to 0 to 1.
mask_dataset = mask_dataset /255.  #PIxel values will be 0 or 1

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.20, random_state = 42)

def trainAndSaveModel():
    from keras.models import Model
    from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization
    from keras.optimizers import Adam
    from keras.layers import Activation, MaxPool2D, Concatenate

    # Building Unet by dividing encoder and decoder into blocks
    def conv_block(input, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(input)
        x = BatchNormalization()(x)   #Not in the original network. 
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)  #Not in the original network
        x = Activation("relu")(x)

        return x

    #Encoder block: Conv block followed by maxpooling


    def encoder_block(input, num_filters):
        x = conv_block(input, num_filters)
        p = MaxPool2D((2, 2))(x)
        return x, p   

    #Decoder block
    #skip features gets input from encoder for concatenation

    def decoder_block(input, skip_features, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
        x = Concatenate()([x, skip_features])
        x = conv_block(x, num_filters)
        return x

    #Build Unet using the blocks
    def build_unet(input_shape, n_classes):
        inputs = Input(input_shape)

        s1, p1 = encoder_block(inputs, 64)
        s2, p2 = encoder_block(p1, 128)
        s3, p3 = encoder_block(p2, 256)
        s4, p4 = encoder_block(p3, 512)

        b1 = conv_block(p4, 1024) #Bridge

        d1 = decoder_block(b1, s4, 512)
        d2 = decoder_block(d1, s3, 256)
        d3 = decoder_block(d2, s2, 128)
        d4 = decoder_block(d3, s1, 64)

        if n_classes == 1:  #Binary
            activation = 'sigmoid'
        else:
            activation = 'softmax'

        outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)  #Change the activation based on n_classes
        print(activation)

        model = Model(inputs, outputs, name="U-Net")
        return model

    IMG_HEIGHT = image_dataset.shape[1]
    IMG_WIDTH  = image_dataset.shape[2]
    IMG_CHANNELS = image_dataset.shape[3]

    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    # this is where the model is built
    model = build_unet(input_shape, n_classes=1)
    model.compile(optimizer=Adam(learning_rate = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, 
                        batch_size = 16, 
                        verbose=1, 
                        epochs=25, 
                        validation_data=(X_test, y_test), 
                        shuffle=False)

    #Save the model for future use
    model.save('cropRowData_25epochs.hdf5')

from keras.models import load_model
model = load_model("cropRowData_25epochs.hdf5", compile=False)
from keras.metrics import MeanIoU
import random

#IOU
# y_pred=model.predict(X_test)
# y_pred_thresholded = y_pred > 0.5

# n_classes = 2
# IOU_keras = MeanIoU(num_classes=n_classes)  
# IOU_keras.update_state(y_pred_thresholded, y_test)
# print("Mean IoU =", IOU_keras.result().numpy())




threshold = 0.5


def testPredictOnThreshhold(threshold):
    test_img_number = random.randint(0, len(X_test)-1)
    test_img = X_test[test_img_number]
    ground_truth=y_test[test_img_number]
    test_img_input=np.expand_dims(test_img, 0)
    print(test_img_input.shape)
    prediction = (model.predict(test_img_input)[0,:,:,0] > threshold).astype(np.uint8)

    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.subplot(233)
    plt.title('Prediction on test image'+str(threshold))
    plt.imshow(prediction, cmap='gray')

# testPredictOnThreshhold(threshold)


def testOnTestImages(threshold):
    import csv
    testIDS = [6, 11, 16, 17, 18, 20, 34, 35, 37, 39, 41, 50, 60, 61, 65, 66, 67, 68, 74, 75, 77, 78, 80, 84, 87, 88, 92, 94, 100, 109, 110, 120, 121, 123, 127, 132, 133, 135, 138, 156, 161, 170, 175, 176, 177, 178, 182, 184, 186, 199, 200, 203, 205, 207, 208, 209, 218, 228, 229, 239, 240, 243, 249, 256, 257, 261, 265, 266, 271, 274, 276]
    # ADD 1 TO EACH ID WHEN SAVING TO CSV

    num_images = len(testIDS)

    image_names = glob.glob("test_images/*.jpg")
    image_names.sort()
    image_names_subset = image_names[0:num_images]
    images = [cv2.imread(img, 0) for img in image_names_subset]
    image_dataset = np.array(images)
    image_dataset = np.expand_dims(image_dataset, axis = 3)
    image_dataset / 255 # normalize between 0 and 1
    
    with open('sample_submission_'+str(threshold)+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ids", "labels"])

        counter = 0
        for img in image_dataset:
            test_img_input=np.expand_dims(img, 0)

            prediction = (model.predict(test_img_input)[0,:,:,0] > threshold).astype(np.uint8)

            # RLE encoding code from announcement
            pixels = (prediction * 255).flatten()
            pixels = np.concatenate([[0], pixels, [0]])
            runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            
            writer.writerow([testIDS[counter]+1, ' '.join(str(x) for x in runs)])
            counter += 1

# testOnTestImages(threshold)
testOnTestImages(threshold-0.1)
testOnTestImages(threshold-0.2)
