import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    
    # Define the approximations for 0 and 1
    oneApprox = 0.99
    zeroApprox = 0.01

    # load in input_data.csv
    df = pd.read_csv('input_data.csv')

    # Get the voltage data
    column_names = df.filter(regex='sensor_\d+').columns
    voltages = df[column_names].values

    # Normalize the voltages by dividing by 2e4
    voltages = voltages / 2e4

    outputImages = np.zeros((len(df), 60, 80))

    for i, row in df.iterrows():
        # Get the filename
        imageFilename = row['filepath']
        
        imageFilePath = os.path.join('./images/processed/60x80', imageFilename)

        # Read the .png file
        image = plt.imread(imageFilePath)

        # Average the four channels
        image = np.mean(image, axis=2)

        # Find all the zero values and replace them with 0.01
        zero_values = image == 0
        image[zero_values] = zeroApprox

        # Find all the one values nd replace them with 0.99
        one_values = image == 1
        image[one_values] = oneApprox

        # Add the image to the outputImages array
        outputImages[i] = image


    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((120, 1), input_shape=(120,)),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(60 * 80, activation='sigmoid'),
        tf.keras.layers.Reshape((60, 80))
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy')

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
    test_image = test_image.reshape(1, 120)
    predicted_image = model.predict(test_image)
    plt.imshow(predicted_image[0], cmap='gray')
    plt.show()



if __name__ == '__main__':
    main()

    
