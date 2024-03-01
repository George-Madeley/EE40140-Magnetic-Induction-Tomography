import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def main(downScaleFactor=1, approximations=(0.01, 0.99)):
    # Define the original width and height
    originalWidth = 480
    originalHeight = 640

    # Define the new width and height
    newWidth = originalWidth // downScaleFactor
    newHeight = originalHeight // downScaleFactor

    # load in input_data.csv
    df = pd.read_csv('data.csv')

    # Get the background voltage and the sensor voltage columns
    bb_names = df.filter(regex='bb_\d+').columns
    cc_names = df.filter(regex='cc_\d+').columns
    # combine the two lists and return the result
    column_names = bb_names.append(cc_names)

    voltages = df[column_names].values

    # get the number of columns in the voltages array
    num_columns = voltages.shape[1]

    # Normalize the voltages by dividing by 2e4
    voltages = voltages / 2e4

    outputImages = np.zeros((len(df), 60, 80))

    for i, row in df.iterrows():
        # Get the filename
        imageFilename = row['cc_filename']
        
        imageFilePath = os.path.join(f'./images/processed/{newWidth}x{newHeight}', imageFilename)

        # Read the .png file
        image = plt.imread(imageFilePath)

        # Average the first three channels
        image = np.mean(image[:, :, :3], axis=2)

        # Find all the zero values and replace them with 0.01
        zero_values = image == 0
        image[zero_values] = approximations[0]

        # Find all the one values nd replace them with 0.99
        one_values = image == 1
        image[one_values] = approximations[1]

        # Add the image to the outputImages array
        outputImages[i] = image


    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((num_columns, 1), input_shape=(num_columns,)),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(newHeight * newWidth, activation='sigmoid'),
        tf.keras.layers.Reshape((newWidth, newHeight))
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        voltages,
        outputImages,
        epochs=1,
        verbose=1,
        validation_split=0.2,
        batch_size=10
    )

    # Test the model
    test_loss = model.evaluate(voltages, outputImages)
    print(f'Test loss: {test_loss}')

    # run the model using the last row of the input data
    test_image = voltages[-1]
    test_image = test_image.reshape(1, num_columns)
    predicted_image = model.predict(test_image)
    plt.imshow(predicted_image[0], cmap='gray')
    plt.show()



if __name__ == '__main__':
    main(downScaleFactor=8, approximations=(0.01, 0.99))

    
