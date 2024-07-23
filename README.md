## Gen high-resolution landscape images from sketch images
### Description
This project was built based on _**pix2pix**_ model and default size of input and output of model is (1024, 1024).
### Process data for training
- Put your dataset (real-landscape-images) for _**dataset/real_photos**_ dir and run _**process_data.py**_ for generate _**dataset/sketch_photos**_, _**dataset/train**_m _**dataset/val**_ dirs.
### Train
- Run _**train.py**_ (default 40000 steps) after have _**train**_ and _**val**_ data.
- Or run script:
  ```sh
  python train.py -e EPOCHS -b BATCH_SIZE -pg PRETRAINED_GEN_PATH -pd PRETRAINED_DISC_PATH \
                  -sg SAVE_GEN_PATH -sd SAVE_DISC_PATH
  ```
### Test
- Run script:
  ```sh
  python inference.py -w WEIGHTS_GEN_PATH -i INPUT_IMG_PATH -o OUTPUT_IMG_PATH
  ```
- Or test code:
  ```sh 
  from inference import gen_image 
  inp_path = INPUT_IMG_PATH 
  out_path = OUTPUT_IMG_PATH 
  gen_image(inp_path, out_path)
  ```
 ### Pretrained
 - You can download pretrained model (train with 2000 landscape images) for test in here:
 _https://drive.google.com/file/d/1nJ17oppifeY5CDYZ21utLXO7P737yto8/view?usp=sharing_
