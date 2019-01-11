This is the code for the paper
"Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition",
Hui Li*, Peng Wang*, Chunhua Shen, Guyu Zhang(* equal contribution) 
published in AAAI-19

1. Installation

The model is implemented in Torch, and has been tested under Ubuntu 14.04, with CUDA 8.0 and CUDNN 7.0.
It depends on the following packages: torch/torch7, torch/nn, torch/nngraph, torch/image, lua-cjson, which can be easily install by "luarocks install **". CUDA-enabled GPUs are required. In addition, LMDB is required which can be installed by "apt-get install liblmdb-dev" and "pip install lmdb" in Ubuntu.

2. Pretrained Model
The pretrained model is localated in https://pan.baidu.com/s/1Z4a0l6UNhuWY3BDy8Z4Ctg because of the space limitation. Download it and put it into the "saved_model" folder.


3. Run the model
To run the model on a new image or image directory, use the script "run_model.lua". 

To run the pretrained model on a provided image, use the '-input_image' flag, for example,
	th run_model.lua -input_image data/beach.jpg
To test the model on an entire directory of images, use the '-input_dir' flag instead:
	th run_model.lua -input_dir /path/to/my/image/folder
The results will be wroten into the folder vis/data.


4. Model training
To train the model, follow the following steps:

4.1. Prepare the training data, including the public available synthetic data:

        Syn90k(http://www.robots.ox.ac.uk/~vgg/data/text/)
	
        SynthText(http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
	
        SynthAdd(https://pan.baidu.com/s/1uV0LtoNmcxbO-0YA7Ch4dg  (code:627x))
   
   
and public available real image datasets:

        IIIT5K (http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)
	
        SVT (http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)
	
        ICDAR2013, ICDAR2015, and COCO-Text (http://rrc.cvc.uab.es/?com=introduction)

4.2. Use the script "create_dataset.py" to generate a group of "data.mdb" files which contain both synthetic and real data. The generated "data.mdb" will be saved under "DataDB" folder. To use create_dataset.py, the training images and their labels should be placed in the imagePathDir and a 'txt' labelfile separately.

4.3. Run the script "th main_train.lua" to train the model. The model will be saved regularly under the folder "saved_model".

Citation
Please cite the following paper if you are using the code/model in your research paper.
@InProceedings{SAR_aaai19,
	author = {Hui Li and Peng Wang and Chunhua Shen and Guyu Zhang},
	title = {Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition},
	booktitle = Proceeding of National Conference on Artificial Intelligence,
	year = {2019},
} 


License

This code is only for academic purpose. For commercial purpose, please contact us.
