# Hazardous object detection for Visually Impaired People

- Install all the dependencies using `pip install -r requirements.txt`
- The `data` folder in root should contain `dataset` folder with raw images. You can run `preprocess.py` to create `processed` folder with preprocessed images.
- To create the YoLov5 model, run:
  ```
  python3 train.py --data data.yaml --epochs 300 --batch-size 16 --img-size 640
  ```
- If you change number or name of classes in your dataset, you need to change `data.yaml` file accordingly.
  