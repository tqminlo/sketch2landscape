## Gen high-resolution landscape images from sketch images
### Description
This project was built based on _**pix2pix**_ model and default size of input and output image is (1024, 1024).
### Process data for training
- Put your dataset (real-landscape-images) for _**real_photos**_ dir and run _**process_data.py**_.
### Train
- Run _**train.py**_ (default 40000 steps) after have _**train_1024**_ and _**val_1024**_ data.
- Checkpoints will be saved in _**training_checkpoint_1024**_ dir.
### Test
- Fix inp_path (path of the sketch-input image) and out_path (path to save the landscape-output image) in _**gen_result.py**_ and then run this file, or you can run this block:
  ```sh 
  from gen_result import gen_image 
  inp_path = YOUR_INP_PATH 
  out_path = YOUR_OUT_PATH 
  gen_image(inp_path, out_path)
  ```
 ### Pretrained
 - You can download pretrained model (train with 40000 landscape images) for test in here:
 _https://drive.google.com/drive/folders/1KKrJlu0DzK5ro7lxNQIH-QsjR59GrZeo?usp=sharing_
