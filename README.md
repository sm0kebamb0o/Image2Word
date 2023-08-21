# Image2Word
Simple Handwritten Text Recognition system implemented using PyTorch. Model could properly work on both GPU and CPU. Accuracy of the recognition is about 70% (fully recognized words) and character_error_rate is lower than 10%, however this parameters depends on the decoding method you choose.

## Attention
Work has not been done yet, so don't judge the quality :grin:

## Training
In order to train the model you should make some initial setup:
* Download IAM dataset with words and store it in a directory called "data\words";
* Save "words.txt" file (that you would find at the same web-page) at the same directory;
* Clone [deslant model](https://github.com/githubharald/DeslantImg);
* Run preprocessing.py;

Now you are ready to use this __awesome__ network :wink:

## Command line arguments
Necessary (one of them):
* `train` <**required number of epochs in training**>
* `predict` <**relative path to the required image**>
* `eval`

Optional (should be used only for testing):
* `-n`, `--new` <**should be used when testing your own images**>
* `-b`, `--beam_width` <**required beam width in decoding**>

* `-lm`, `--lm_inluence` <**required language model influence in decoding $\in$ (0; 1]**>