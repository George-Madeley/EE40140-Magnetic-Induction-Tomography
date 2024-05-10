from matplotlib import pyplot as plt
import pandas as pd

import os

def plot_before_and_after(
    down_scale_factor = 1,
    font_size = 20,
    verbose = False
):
  
  after_width = 640 // down_scale_factor
  after_height = 480 // down_scale_factor

  file_path = os.path.join('data', 'data_samples.csv')
  df = pd.read_csv(file_path)

  # Get a list of the unique values in the 'sample' column
  samples = df['sample'].unique()

  for sample in samples:
    # get a random record from the dataframe where the 'sample' column is equal
    # to the current sample
    sample_df = df[df['sample'] == sample].sample(1)

    # get the cc_filename from the record
    cc_filename = sample_df['cc_filename'].values[0]

    before_img = os.path.join('images', 'original', cc_filename)
    after_img = os.path.join('images', 'processed', f'{after_height}x{after_width}', cc_filename)

    before_img = plt.imread(before_img)
    after_img = plt.imread(after_img)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(before_img)
    ax[0].set_title('Before')
    ax[0].axis('off')
    
    ax[1].imshow(after_img)
    ax[1].set_title('After')
    ax[1].axis('off')

    # add a title to the figure
    fig.suptitle(f'Before and After for {sample}', fontsize=font_size)

    plt.tight_layout()

    if verbose:
      plt.show()

    save_path = os.path.join('images', 'graphs', 'before and after')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{sample}.png'))

  
if __name__ == '__main__':
  plot_before_and_after()