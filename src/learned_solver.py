import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from cvae import SamplerCVAE

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="forest", help="Name of the environment to train/test the model (shifting_gaps, forest, maze etc)")
    parser.add_argument("--train", type=bool, default=False, help="Select to train the model")
    parser.add_argument("--test", type=bool, default=False, help="Select to test the model")
    parser.add_argument("--num_test_samples", type=int, default=10, help="Number of test samples to test the model")
    parser.add_argument("--model_location", type=str, default=None,help="Give path to the model to test")
    args = parser.parse_args()
    env_name = args.env_name
    dataset_parent_dir = "./../motion_planning_datasets/"
    dataset_dir = dataset_parent_dir + env_name
    # print the folder name and the folders in the directory
    print("dataset_dir: ", dataset_dir)
    print("folders in the env_dir: ", os.listdir(dataset_dir))
    # Check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        raise ValueError("Dataset directory does not exist")
    # Get the folder's images
    train_folder_images = os.listdir(dataset_dir + "/train")
    print("Number of images in the train folder: ", len(train_folder_images))
    val_folder_images = os.listdir(dataset_dir + "/validation")
    print("Number of images in the validation folder: ", len(val_folder_images))
    test_folder_images = os.listdir(dataset_dir + "/test")
    print("Number of images in the test folder: ", len(test_folder_images))
    all_data = (train_folder_images, 
                val_folder_images, 
                test_folder_images)


    print("Env_name: ", env_name)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = SamplerCVAE(all_data,
                        device, 
                        train=args.train,
                        test=args.test,
                        env_name=env_name)
    print("Model: ", model)
    model.to(device)
    
    if args.train:
        print("Training the model")
        model.fit()
    elif args.test:
        if args.model_location == None:
            raise ValueError("Please specify the model path")
        # Check if model path exists
        if not os.path.exists(args.model_location):
            raise ValueError("Model path does not exist")
        # Check if dataset folder exists
        if not os.path.exists(dataset_dir):
            raise ValueError("Dataset folder does not exist")
        print("Dataset Path: ", dataset_dir)
        print("Model Path: ", args.model_location)
        model.test(args.model_location,start_point=(1,1),
                    goal_point=(99,99),
                    env_name=env_name,
                    num_test_samples=args.num_test_samples)
    else:
        raise ValueError("Please specify either --train or --test")
    print("Done.")
    
    # To test the model
    # python learned_solver.py --test True --model_location forest_epoch_570-0.01331585762090981.pt