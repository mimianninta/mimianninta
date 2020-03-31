# Object Detection

This repository contains a tutorial of Tensorflow Object Detection using custom dataset.

# Prepare data






# Setup the environment


1. git clone
2. goto detection & run commands for reqired installation

```
!curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip
!unzip protoc-3.2.0-linux-x86_64.zip -d protoc3
!mv protoc3/bin/* /usr/local/bin/
!mv protoc3/include/* /usr/local/include/
```

```
# install pycocotool for evaluation
!apt-get update
!apt install -y git python3-tk libsm6 libxext6
!pip3 install cython pillow opencv-python
!pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```


```
!sudo pip3 install -r requirements1.txt
```

4. goto detection/models/research and run

```
!protoc object_detection/protos/*.proto --python_out=. # protoc needs to be version 3
!export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim # needs to be executed each time in a new shell
```

# Data Preparation


## 1. Prepare TFRecord

In order to train a tensorflow model, we need to prepare the data in its acceptable form, which are `tfrecord`s.

Goto detection & Run

```
!python create_tfrecords.py --output_path data/*_train.record --csv_input data/*_train.csv --image_path data/images/train/
```

```
!python create_tfrecords.py --output_path data/*_val.record --csv_input data/*_val.csv --image_path data/images/valid/
```

with the paths set correctly to your paths.

## 2. Create the label map

From `detection/data/fish_label_map.pbtxt`, you can see sample label maps.

Create your own according to your classes.

**Note** : The class id must start from **1**.

Now the data prepartion is completed. We move on to prepare the model.


## 3. Pick the corresponding config

Pick the model's config from `/detection/models/research/object_detection/samples/configs`. Duplicate it somewhere.

**Note** : you must pick the config with the **same** name as your model.

Or Modify the config from `detection/faster_rcnn_inception_v2_coco.config`

* `num_classes`, which should be at the beginning of the file;
* `fine_tune_checkpoint`, path to weight file folder;
* `num_steps`;
* `batch_size` and
* 
```
train_input_reader: {
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record"
  }
  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
}

eval_config: {
  num_examples: 8000
  # Note: The below line limits the evaluation process to 10 evaluations.
  # Remove the below line to evaluate indefinitely.
  max_evals: 10
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record"
  }
  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
  shuffle: false
  num_readers: 1
}
```
which are towards the end of the file.

**Note** : The paths should be **absolute**!



# Training

From `detection/models/research/`, run
```
!python train.py --logtostderr --train_dir=${YOUR MODEL'S OUTPUT DIR} --pipeline_config_path=${YOUR CONFIG's PATH} 
```
**Note** : The paths should be **absolute**!

You will get the following training info :

```
...
INFO:tensorflow:global step 1794: loss = 0.5385 (8.267 sec/step)
...
```

You can run `tensorboard --logdir=${YOUR MODEL'S OUTPUT DIR}` to check if the loss actually decreases.



# Evaluation

From `detection/models/research/`, run
```
!python eval.py --logtostderr --checkpoint_dir=${YOUR MODEL'S OUTPUT DIR} --pipeline_config_path=${YOUR CONFIG's PATH} --eval_dir=${YOUR EVAL'S OUTUPT DIR} 
```

Then, run `tensorboard --logdir=${YOUR EVAL'S OUTUPT DIR}`. 

You should see some validation images like the following :
![eval](images/eval.png)

# Export the graph

Once your model is trained, you need to export a `.pb` graph, to use for inference.

From `detection/models/research/`, run
```
!python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path ${YOUR CONFIG's PATH} --trained_checkpoint_prefix ${YOUR MODEL'S OUTPUT DIR}/model.ckpt-XXXX --output_directory ${YOUR GRAPH's PATH}
```
where `XXXX` is the last checkpoint step (the largest number in that folder).


# Run inference

1. First, find some images with objects you want to detect inside. Download or keep them to `detection/models/research/object_detection/` with format `.jpg`.

2. Move the folder containing the `.pb` graph to `/detection/models/research/` and rename the folder as `inference_graph`.

3. Then copy the label map to `/detection/models/research/training/` and rename the file as `labelmap.pbtxt`

4. Open the python file `detection/models/research/Object_detection_image_new.py` and Modify `NUM_CLASSES`.

5. Run command

```
!python Object_detection_image_new.py
```

The output images will be saved in `/detection/models/research/output/` folder.



# References

https://github.com/kwea123/fish_detection.git
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

