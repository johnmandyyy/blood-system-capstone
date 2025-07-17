from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import *
from keras.callbacks import *
from keras.losses import *
from keras import optimizers
from keras.layers import Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from app.constants import app_constants
from sklearn.model_selection import train_test_split
from app.models import Images
from tensorflow.keras.preprocessing import image
from app.models import ModelInfo
import pandas as pd
import cv2
import imutils
import datetime
from tensorflow.keras.regularizers import l2

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model


import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
import gc

class SegmentationDataGenerator(Sequence):

    def __init__(self, file_list, batch_size=8, target_size=(224, 224), is_binary=True):
        self.file_list = file_list
        self.batch_size = batch_size
        self.target_size = target_size
        self.is_binary = is_binary

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.file_list[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        masks = []

        for img_path, mask_path in batch_files:
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.target_size)
            img = img.astype("float32") / 255.0

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.target_size)
            mask = mask.astype("float32") / 255.0

            if self.is_binary:
                mask = np.expand_dims(mask, axis=-1)

            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)
    

# physical_devices = tf.config.list_physical_devices("GPU")
# if physical_devices:
#     tf.config.set_visible_devices([], "GPU")
# matplotlib.use('TkAgg')  # Or 'Agg', 'Qt5Agg' depending on your environment

class GradCAM:

    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output.shape) == 4:
                return layer.name

        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output],
        )

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)

class CNN:

    def __init__(self):
        
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.accuracy = 0
        self.conf_matrix = []
        self.class_names = []
        self.json_info = {}
        self.heatmap_location = ''
        self.severity = ''
        self.segmented_location = ''
        self.original_location = ''

    def get_severity(self):
        """ Get severity percentage. """
        return self.severity

    def get_heatmap_location(self):
        """ Get heatmap location after prediction. """
        return self.heatmap_location
    
    def get_original_image(self):
        """ Get the non enhanced image. """

        if self.original_location == '':
            return None
        return "/media/patient_smears/" + self.original_location

    def get_indices(self):
        """ Gets the indices of class names. """

        model_info = ModelInfo.objects.all()
        if len(model_info) > 0:
            for info in model_info:
                json_indices = json.loads(info.json_info)
                return json_indices['indices']
            
    def load_segmentor_model(self, smear_image_location):
        """ Loads the segmented model and predict using the trained Xception model. """

        location = 'media/' + str(smear_image_location)

        np_frame = cv2.imread(location)
        if np_frame is None:
            raise FileNotFoundError(f"Error: Unable to load image at {location}")
        
        model = load_model(app_constants.SEG_MODEL_NAME)
        resized_image = cv2.resize(np_frame, (224, 224))
        img_array = image.img_to_array(resized_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # --- Predict Mask ---
        pred_mask = model.predict(img_array)[0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        # --- Create Cutout ---
        mask_3ch = np.stack([pred_mask.squeeze()]*3, axis=-1)
        cutout = cv2.bitwise_and(resized_image, mask_3ch)

        fname = datetime.datetime.now().strftime("output_%Y%m%d_%H%M%S.jpg")

        # --- Save Cutout ---
        location = 'media/' + str(smear_image_location) + "_segmented" + fname
        cv2.imwrite(location, cutout)
        self.segmented_location = location

        del model
        K.clear_session()
        gc.collect()

        return location
    
    def get_segmented_image(self):
        """ Gets the segmented image. """

        if self.segmented_location == '':
            return None
        
        return "/" + self.segmented_location

    def predict_image_v2(self, smear_image_location):
        """Loads the model and predicts using the trained Xception model, applying heatmap on original image."""
        
        original_image = cv2.imread(str(smear_image_location.file))
        enhanced_image = self.enhance_image(original_image, str(smear_image_location.file))
        
        self.segmented_location = '' # Initialize empty location first.
        list_of_class_names = self.get_indices()
        
        # Load segmented image for prediction
        segmented_location = self.load_segmentor_model(smear_image_location)
        segmented_image = cv2.imread(segmented_location)
        if segmented_image is None:
            raise FileNotFoundError(f"Error: Unable to load segmented image at {segmented_location}")
        
        # Load original image for heatmap overlay
        enhanced_image = cv2.imread(str(smear_image_location.file))
        if enhanced_image is None:
            raise FileNotFoundError(f"Error: Unable to load original image at {smear_image_location}")

        # Resize segmented image for prediction
        resized_image = cv2.resize(segmented_image, (224, 224))
        img_array = image.img_to_array(resized_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Load model and predict
        model = load_model(app_constants.MODEL_NAME)
        prediction = model.predict(img_array)

        # Get predicted class
        highestProb = np.argmax(prediction)
        predicted_class_name = list_of_class_names[highestProb]
        confidence_percentage = prediction[0][highestProb] * 100
        self.severity = round(confidence_percentage, 2)

        # GradCAM on original image
        cam = GradCAM(model, highestProb)
        prep = preprocess_input(np.expand_dims(cv2.resize(enhanced_image, (224, 224)), axis=0))
        heatmap = cam.compute_heatmap(prep)
        heatmap = cv2.resize(heatmap, (enhanced_image.shape[1], enhanced_image.shape[0]))  # match original size
        (heatmap, output) = cam.overlay_heatmap(heatmap, enhanced_image, alpha=0.5)

        output = imutils.resize(output, height=700)
        filename = datetime.datetime.now().strftime("output_%Y%m%d_%H%M%S.jpg")
        relative_path = "media/generated_heatmap/" + filename
        cv2.imwrite(relative_path, output)
        self.heatmap_location = relative_path

        del model
        K.clear_session()
        gc.collect()
        
        return predicted_class_name

    def enhance_image(self, image, save_path=None):
        """This method sharpens images for clarity.
        It saves the original image with 'original_' prefix before enhancement.
        """

        # Save the original image (before enhancement)
        if save_path is not None:
            dir_name = os.path.dirname(save_path)
            base_name = os.path.basename(save_path)
            original_name = "original_" + base_name
            original_path = os.path.join(dir_name, original_name)
            self.original_location = original_name
            cv2.imwrite(original_path, image)  # Save the unprocessed original
            print(f"Original image saved at: {original_path}")

        # Enhance the image
        img = cv2.convertScaleAbs(image, alpha=1.1, beta=-20)
        sharpen_amount = 1.1
        if sharpen_amount > 0:
            alpha = sharpen_amount
            kernel = np.array([[0, -alpha, 0],
                            [-alpha, 1 + 4 * alpha, -alpha],
                            [0, -alpha, 0]])
            img = cv2.filter2D(img, -1, kernel)

        # Save enhanced image (overwrite original if needed)
        if save_path is not None:
            cv2.imwrite(save_path, img)
            print(f"Enhanced image saved at: {save_path}")

        return img

    def predict_image(self, smear_image_location):
        """ Loads the model and predict using the trained Xception model. """
        
        list_of_class_names = self.get_indices()
        segmented_location = self.load_segmentor_model(smear_image_location)
        location = smear_image_location

        np_frame = cv2.imread(segmented_location) # Read the segmented image.
        if np_frame is None:
            raise FileNotFoundError(f"Error: Unable to load image at {location}")
        
        model = load_model(app_constants.MODEL_NAME)
        resized_image = cv2.resize(np_frame, (224, 224))
        img_array = image.img_to_array(resized_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        prediction = model.predict(img_array)

        highestProb = np.argmax(prediction)
        cam = GradCAM(model, highestProb)
        prep = preprocess_input(np.expand_dims(np_frame, axis=0))
        heatmap = cam.compute_heatmap(prep)
        heatmap = cv2.resize(heatmap, (224, 224))
        (heatmap, output) = cam.overlay_heatmap(heatmap, np_frame, alpha=0.5)

        font_size = 0.5

        # Show the image.
        cv2.putText(
            np_frame,
            f"Predicted: {list_of_class_names[highestProb]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (0, 0, 255),
            2,
        )

        confidence_percentage = prediction[0][highestProb] * 100 
        self.severity = round(confidence_percentage, 2)

        output = np.vstack([output])
        output = imutils.resize(output, height=700)
        filename = datetime.datetime.now().strftime("output_%Y%m%d_%H%M%S.jpg")
        relative_path = "media/generated_heatmap/" + filename

        cv2.imwrite(relative_path, output)

        self.heatmap_location = relative_path

        d = prediction.flatten()
        j = d.max()
        for index, item in enumerate(d):
            if item == j:
                predicted_class_name = list_of_class_names[index]
                return predicted_class_name

    def create_model_layers(self):
        """ Create base model layers for CNN. """

        xception_instance = Xception(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )

        # Build the model
        # x = Flatten()(xception_instance.output)
        # x = Dense(128)(x)
        # x = BatchNormalization()(x)  # Batch normalization before activation
        # x = Activation("relu")(x)  # ReLU activation
        # x = Dropout(0.1)(x)  # Add dropout for regularization

        xception_instance.trainable = False

        # Add custom classification head
        x = GlobalAveragePooling2D()(xception_instance.output)  # Better than Flatten
        x = Dense(256, kernel_regularizer=l2(0.0001))(x)  # L2 regularization
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.3)(x)  # Increased dropout for better generalization

        predictions = Dense(4, activation="softmax")(x)  # Final output layer
        model = Model(inputs=xception_instance.input, outputs = predictions)
        model.summary()

        return model
        
    def retrieve_files_from_database(self):
        """ Get data from database. """

        files = Images.objects.all()
        extended_file = []
        
        for image in files:
            absolute_path = "media/" + str(image.location)
            class_name = image.disease.disease_name
           
            extended_file.append(
                (absolute_path, class_name)
            )

        return extended_file
    
    def mark_images(self, training_images, validation_images):
        """ Used to mark images and record the training and validation images used. """

        Images.objects.all().update(used_for = 'No Remark') # Set as No Remark First

        for file, label in training_images:
            find_this_file = str(file).replace("media/", "")
            Images.objects.all().filter(fname = find_this_file).update(used_for = 'Training')

        for file, label in validation_images:
            find_this_file = str(file).replace("media/", "")
            Images.objects.all().filter(fname = find_this_file).update(used_for = 'Validation')
        
    
    def generate_splits(self):
        """ Generate data splits. """

        # Paths and Constants
        BATCH_SIZE = 16
        TARGET_SIZE = (224, 224)

        all_files = self.retrieve_files_from_database()
        
        # Split into train and validation (80% train, 20% validation)
        train_files, val_files = train_test_split(all_files, test_size=0.2, stratify=[label for _, label in all_files])

        self.mark_images(train_files, val_files) # Record to Database

        # Data generators
        train_datagen = ImageDataGenerator(rescale=1./255)
        valid_datagen = ImageDataGenerator(rescale=1./255)

        # Create train and validation generators
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame(train_files, columns=["filename", "class"]),
            x_col="filename",
            y_col="class",
            target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        valid_generator = valid_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame(val_files, columns=["filename", "class"]),
            x_col="filename",
            y_col="class",
            target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        return train_generator, valid_generator


    def train_images(self):
        """ Train images from datasets. """

        model = self.create_model_layers()
        
        opi = optimizers.Adam(learning_rate=1e-4)
        model.compile(
            optimizer=opi, loss=categorical_crossentropy, metrics=["accuracy"]
        )

        train_generator, valid_generator = self.generate_splits()
        batch_size = 16

        class_indices = train_generator.class_indices
        class_names = list(class_indices.keys())

        if True:

            reduce_lr = ReduceLROnPlateau(
                monitor="loss",
                factor=0.1,  # Factor by which the learning rate will be reduced. New_lr = lr * factor
                patience=3,  # Number of epochs with no improvement after which learning rate will be reduced.
                verbose=1,  # 0: quiet, 1: update messages
                mode="min",  # In mode 'min', lr will be reduced when the quantity monitored has stopped decreasing
                min_delta=0.001,  # Minimum change in the monitored quantity to qualify as an improvement
                cooldown=0,  # Number of epochs to wait before resuming normal operation after lr has been reduced.
                min_lr=0,  # Lower bound on the learning rate.
            )

            checkpoint = ModelCheckpoint(
                filepath=app_constants.MODEL_NAME,  # Path where the model will be saved
                monitor="val_accuracy",  # Metric to monitor
                save_best_only=True,  # Save only the best model
                mode="max",  # Mode: 'min' for minimizing the monitored metric
                verbose=1,  # Verbosity mode
            )

            early = EarlyStopping(
                monitor="val_accuracy", min_delta=0, patience=20, verbose=1, mode="auto"
            )

            history = model.fit_generator(
                steps_per_epoch=train_generator.samples // batch_size,
                generator=train_generator,
                validation_data=valid_generator,
                validation_steps=valid_generator.samples // batch_size,
                epochs=app_constants.EPOCHS,
                callbacks=[checkpoint, early, reduce_lr],
            )

            model = load_model(app_constants.MODEL_NAME)
            evaluation = model.evaluate(valid_generator)
            print("Test Accuracy:", evaluation[1], evaluation)

            preds = model.predict(valid_generator)
            predicted_classes = np.argmax(preds, axis=1)

            # Get the actual labels
            true_classes = valid_generator.classes
            filenames = valid_generator.filenames

            # model.save('fmodel.h5')

            # Determine correctness
            # results = [
            #     f"{filename} - {'CORRECT' if pred == true else 'INCORRECT'}"
            #     for filename, pred, true in zip(filenames, predicted_classes, true_classes)
            # ]

            results = []
            
            for filename, predictions, remarks in zip(filenames, predicted_classes, true_classes):
                print(predictions, remarks)

                fremarks = 'CORRECT' if predictions == remarks else 'INCORRECT'
                results.append(
                    {
                        "remark": fremarks,
                        "filename": filename
                    }
                )
            #import json
            #print(json.dumps(results))

            # Calculate the confusion matrix
            self.conf_matrix = confusion_matrix(true_classes, predicted_classes)
            self.class_names = class_names # UP TO DOWN AND LEFT TO RIGHT IF PRINTED
            self.accuracy = round(accuracy_score(true_classes, predicted_classes), 2)

            self.precision = round(precision_score(
                true_classes, predicted_classes, average="weighted"
            ), 2)
            
            self.recall = round(recall_score(true_classes, predicted_classes, average="weighted"), 2)
            self.f1_score = round(f1_score(true_classes, predicted_classes, average="weighted"), 2)

            self.json_info = {
                'data': self.conf_matrix.tolist(),
                'indices': self.class_names,
                'remarks': results
            }

