# Crop Row Segmentation

## Creating Datasets

First I sorted the images that had masks into a folder. I then used the `glob` library to grab those images and create a group with the `cv2.imread()` function. That was then processed into a dataset using the `np.array()` and `np.expand_dims()` functions respectively.

The same process was repeated for the masks, which I had converted from numpy arrays into images.

```python
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
```

I assert that the shape of the image_dataset and mask_dataset are the same, and normalize the images.

```python
print("Image data shape is: ", image_dataset.shape)
print("Mask data shape is: ", mask_dataset.shape)

#Normalize images
image_dataset = image_dataset /255. 
mask_dataset = mask_dataset /255.  #Pixel values will be 0 or 1
```

## Building the Unet Model

I define a convolution block as preforming Conv2D, BatchNormalization, and Activation operations from the `keras.layers` library. This block will be used multiple times in the model.

```python
# Building Unet by dividing encoder and decoder into blocks
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x
```

I define a encoder block by just preforming a convolution block and then regularize with maxpooling.

```python
#Encoder block: Conv block followed by maxpooling
def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p   
```

I now define a decoder block to preform an inverse type of operation compared to the encoder block. I preform a Conv2DTranspose operation and skip features based on the output from the pooling in the encoder block.

```python
#Decoder block
#skip features gets input from encoder for concatenation
def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x
```

I can now build a unet model using these blocks, passing graduately larger filter numbers. Through a bridge convolution, I then decode the weights through the filter numbers backwards. The outputs are then Conv2Ded with a **sigmoid activation** as the segmentation is *binary*. I then create a keras model named 'U-Net'.

```python
#Build Unet using the blocks
def build_unet(input_shape):
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

    outputs = Conv2D(1, 1, padding="same", activation='sigmoid')(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model
```

## Training the Model

The program now compliles the Keras model with a binary_crossentropy loss function, graded by accuracy. The model is then fitted with a test_size 20% of the 210 images with matching masks. **I chose to use a batch size of 16 and 25 epochs.** More accurate models might have been possible with further tweaking.

The model is saved as a hdf5 file so I could preform predictions with it without having to retrain a model every time.

```python
# this is where the model is built
model = build_unet((image_dataset.shape[1], image_dataset.shape[2], image_dataset.shape[3]), n_classes=1)
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
```

## Computing Predictions on new Data

I had identified which ids applied to images with no mask. I grabed all of them and created a dataset in a similar procedure to [the datasets above.](#creating-datasets)

```python
num_images = len(testIDS)

image_names = glob.glob("test_images/*.jpg")
image_names.sort()
image_names_subset = image_names[0:num_images]
images = [cv2.imread(img, 0) for img in image_names_subset]
image_dataset = np.array(images)
image_dataset = np.expand_dims(image_dataset, axis = 3)
image_dataset / 255 # normalize between 0 and 1
```

I create a csv file to store these predictions making sure to follow the same format as the `sample_submission.csv`.

```python
with open('sample_submission_'+str(threshold)+'.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ids", "labels"])
```

I then iterate through each image in this dataset, making sure to match the shape the model was trained for. The predictions for the pixels are then compared to a threshold value (0.3~0.5). The operation results in an array of booleans, so the `astype` method for a array converts it into integers 0 and 1.

``` python
counter = 0
for img in image_dataset:
    test_img_input=np.expand_dims(img, 0)

    prediction = (model.predict(test_img_input)[0,:,:,0] > threshold).astype(np.uint8)
```

That prediction array is converted to a color value (0 or 255) then converted into an rle based on the code provided to us. That long string is written into the csv along with the correct id.

```python
# RLE encoding code from announcement
pixels = (prediction * 255).flatten()
pixels = np.concatenate([[0], pixels, [0]])
runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
runs[1::2] -= runs[::2]

writer.writerow([testIDS[counter]+1, ' '.join(str(x) for x in runs)])
counter += 1
```

This prediction computing is ran multiple times on different threshold values, the best 3 were submitted for the assignment.

## Challenges

Initially I tried to take advantage of a [github project](https://github.com/qubvel/segmentation_models.pytorch) that had a built in u-net model. I couldn't figure out how to successfully input the images as a dataset, and the image dimensions had to be a mutltiples of 32 for parallelization reasons. That would have caused my model to fail if I could ever input the data anyway as the height of the images are 240 pixels tall. This was incredibly frustrating as it took so long to reach the point of realization that I couldn't ever get a model from this approach. After all that effort, it was actually faster to learn how to create a U-net model through keras operations.

I believe that there are some optimization techniques that I could have used to improve my model like data augmentation, I have heard from some peers that that method greatly improved results. I unfortunately was out of time.
