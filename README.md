# SimpleHTR now on PyTorch
Simple Handwritten Text Recognition system implemented using PyTorch. Model could properly work on both GPU and CPU. Accuracy of the recognition is about 80% (fully recognized words) and character_error_rate is lower than 10%, however this parameters depends on the decoding method you choose.

## Training
In order to train the model you should make some initial setup:
* Download IAM dataset with words and store it in a directory called "data\words";
* Save "words.txt" file (that you would find at the same web-page) at the same directory;
* Run data_normalizer.py;

Now you are ready to use this __awesome__ network :wink:

## Command line arguments
Necessary (one of them):
* `--train` <**_number_of_epochs_**>
* `--test` <**_relative_path_to_the_image_**>

Optional:
* `--beam_width` <**_value_**>
* `--lm_inluence` <**_value_** $\in$ (0; 1]>