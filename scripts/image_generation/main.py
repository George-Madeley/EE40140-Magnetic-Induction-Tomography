import torch
from torch.optim import Adam
from torch import nn

import pandas as pd

import matplotlib.pyplot as plt


from dataExtraction import dataExtraction
from models.Discriminator import Discriminator
from models.Generator import Generator


def main(downScaleFactor=1, approximations=(0.01, 0.99)):
    # check if cuda is available for training on GPU
    device = ""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # define the batch size
    batchSize = 50

    # calculate the number of batches
    data = pd.read_csv("data.csv")
    numData = len(data)
    numBatches = numData // batchSize

    randomSampleIndeces = [1342, 941, 119, 823, 607, 414]

    # extract data
    trainLoader = dataExtraction(downScaleFactor, approximations)

    width = 640 // downScaleFactor
    height = 480 // downScaleFactor
    
    numInputs = 240

    
    # define the hyperparameters
    learningRate = 0.001
    numEpochs = 1000
    lossFunctionDiscriminator = nn.BCELoss()
    lossFunctionGenerator = nn.L1Loss()

    # Create an instance of the discriminator and generator
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    # define the discriminator and generator optimizers
    optimizerDiscriminator = Adam(discriminator.parameters(), lr=learningRate)
    optimizerGenerator = Adam(generator.parameters(), lr=learningRate)

    for epoch in range(numEpochs):
        for n, (realSamples, latentSpaceSamples) in enumerate(trainLoader):
            # Data for training the discriminator

            # Get the real samples and send to device
            realSamples = realSamples.to(
                device=device
            )

            # Create the labels which are later used as input for the BCE loss
            # function for the discriminator and send to device
            realSampleLabels = torch.ones((batchSize, 1)).to(
                device=device
            )

            # Send the latent samples to the device
            latentSpaceSamples = latentSpaceSamples.to(
                device=device
            )

            # Generate the fake samples from the latent samples
            generatedSamples = generator(latentSpaceSamples)

            # Create the labels for the fake samples
            generatedSampleLabels = torch.zeros((batchSize, 1)).to(
                device=device
            )

            # Combine the real and fake samples
            allSamples = torch.cat((realSamples, generatedSamples))
            allSampleLabels = torch.cat(
                (realSampleLabels, generatedSampleLabels)
            )

            # Training the discriminator
            discriminator.zero_grad()
            outputDiscriminator = discriminator(allSamples)
            lossDiscriminator = lossFunctionDiscriminator(
                outputDiscriminator, allSampleLabels
            )
            lossDiscriminator.backward()
            optimizerDiscriminator.step()

            # Send the latent samples to the device
            latentSpaceSamples = latentSpaceSamples.to(
                device=device
            )

            # Training the generator
            generator.zero_grad()
            generatedSamples = generator(latentSpaceSamples)
            lossGenerator = lossFunctionGenerator(
                generatedSamples, realSamples
            )
            lossGenerator.backward()
            optimizerGenerator.step()

            # Show loss
            if n == numBatches - 1:
                print(f"Epoch: {epoch} Loss D.: {lossDiscriminator}")
                print(f"Epoch: {epoch} Loss G.: {lossGenerator}")
                print("--------------------------------------------------")

        # Generate images using random latent samples
        if epoch == 0:
            fixedLatentSamples = trainLoader.dataset.tensors[1][randomSampleIndeces].to(device)
        generatedImages = generator(fixedLatentSamples)
        generatedImages = generatedImages.detach().cpu()

        # Plot and save the generated images
        fig, axs = plt.subplots(2, 3, figsize=(8, 6))
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(generatedImages[i][0], cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        
        # title the plot
        plt.suptitle(f"Epoch {epoch}")
        # save the plot
        plt.savefig(f"./images/epochs/epoch_{str(epoch).zfill(3)}.png")
        plt.close()

if __name__ == '__main__':
    main(downScaleFactor=8, approximations=(0.01, 0.99))

    
