import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/dataset", help="Directory with the objects images")
parser.add_argument("--output_dir", default="data/cropped", help="Where to write the new dataset")

if __name__ == "__main__":
    args = parser.parse_args()
    
    dataset_dir = args.data_dir
    output_dir = args.output_dir
    
    print(dataset_dir,output_dir)
    
    # for each class in dataset_dir, split into train, dev, test and save in output_dir
    for class_dir in os.listdir(dataset_dir):
        print(class_dir)
        # get all images in class_dir
        class_dir_path = os.path.join(dataset_dir, class_dir)
        images = os.listdir(class_dir_path)
        images = [os.path.join(class_dir_path, f) for f in images if f.endswith(".jpg")]
        print(images)
        # split into train, dev, test
        
        random.seed(230)
        random.shuffle(images)
        split_1 = int(0.8 * len(images))
        split_2 = int(0.9 * len(images))
        train_filenames = images[:split_1]
        dev_filenames = images[split_1:split_2]
        test_filenames = images[split_2:]
        
        filenames = {"train": train_filenames, "dev": dev_filenames, "test": test_filenames}
        
        # create output_dir/class_dir
        output_class_dir = os.path.join(output_dir, class_dir)
        
        if not os.path.exists(output_class_dir):
            os.mkdir(output_class_dir)
        else:
            print("Warning: output dir {} already exists".format(output_class_dir))
            
        # create output_dir/class_dir/train, output_dir/class_dir/dev, output_dir/class_dir/test 
        for split in ["train", "dev", "test"]:
            output_dir_split = os.path.join(output_class_dir, split)
            if not os.path.exists(output_dir_split):
                os.mkdir(output_dir_split)
            else:
                print("Warning: output dir {} already exists".format(output_dir_split))
                
            # copy images from filenames[split] to output_dir_split
            for image in filenames[split]:
                os.system("cp {} {}".format(image, output_dir_split))
                
        # break # for debugging purposes
        
    
