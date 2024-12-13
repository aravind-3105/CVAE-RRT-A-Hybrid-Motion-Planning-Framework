# Standard library imports
import os
import time
import random
from datetime import datetime
import pickle

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.animation import FFMpegWriter
from sklearn.neighbors import KernelDensity as KDE2D
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Local imports
from env_2d import Env2D
from rrtstar2d import RRTstar


def make_video(image_set, pathOut, fps=10):
    """
    Create an MP4 video from a set of images using matplotlib's animation module.

    Args:
        image_set (list): List of numpy arrays (images).
        pathOut (str): Path to save the output video.
        fps (int): Frames per second.
    """
    fig, ax = plt.subplots(figsize=(16,16))
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, pathOut, dpi=150):
        for img in image_set:
            ax.clear()
            ax.imshow(img)
            ax.axis("off")
            writer.grab_frame()
    print(f"Video saved to {pathOut}")

def mplfig_to_npimage(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    image = np.asarray(buf)
    return image[:, :, :3]

class mapsDataset(Dataset):   
    def __init__(self, dataset, device):        
        self.dataset = dataset
        self.device = device
        self.total_data_num = len(self.dataset)

    def __len__(self):
        return self.total_data_num

    def __getitem__(self, idx):
        return self.dataset[idx]

    def rollout(self, samples):
        x_batch = np.stack([sample[0] for sample in samples], axis=0)
        y_batch = np.array([sample[1] for sample in samples])  

        # Convert to tensors
        x_batch = torch.tensor(x_batch, 
                               dtype=torch.float32, 
                               device=self.device).unsqueeze(1) 
        y_batch = torch.tensor(y_batch, 
                               dtype=torch.float32, 
                               device=self.device)

        return x_batch, y_batch

class SamplerCVAE(nn.Module):

    def __init__(self, env_workspaces, device, train=False,test=False,env_name='forest'):
        super(SamplerCVAE, self).__init__() # inherit the __init__ from nn.Module
        # Set random seed
        torch.manual_seed(777)
        torch.cuda.manual_seed(777)
        
        # Global parameters
        self.device = device
        self.x_lims = (0, 100)
        self.y_lims = (0, 100)
        self.env_params = {'x_lims': self.x_lims, 'y_lims': self.y_lims}
        self.env_name = env_name
        self.env_dir = "./../motion_planning_datasets/" + self.env_name
        # Make directories for the model
        # 1. Make a directory for storing the checkpoints
        # 2. Make a directory for storing the training images
        self.time_now = datetime.now().strftime('%Y_%m_%d_%H_%M')
        
        #Common hyperparameters
        self.learning_rate = 0.0001
        self.batch_size = 128
        self.extend_len = 5 # Can be 20 also
        self.train_dir = None
        self.val_dir = None
        self.test_dir = None
        
        # Make workspace
        self.train_workspace, self.val_workspace, self.test_workspace = env_workspaces
        
        
        if train:
            dataset_dir = "./data_collection_pickle/"
            data_type = env_name
            data_path = dataset_dir + data_type + "/" 
            # Print the data path
            print("Data path: ", data_path)
            # That folder has train, validation, and test folders
            # Check if the folder exists
            if not os.path.exists(data_path):
                # Raise error
                raise ValueError("Invalid dataset path")
            # Check if the folder has train, validation, and test folders
            train_folder = data_path + "train/"
            val_folder = data_path + "validation/"
            # Make checkpoints folder
            # Check if the folder exists
            if not os.path.exists("./checkpoints/"):
                os.makedirs("./checkpoints/", exist_ok=True)
            # Make a subfolder with the name env_name_time
            self.checkpoint_dir = os.path.join("./checkpoints/", env_name + "__" + self.time_now)
            os.makedirs(self.checkpoint_dir)
            
            # Hyperparameters
            self.num_epochs = 500
            self.batch_size = 128 # Can increase to 256/512 also based on the GPU.
            self.learning_rate = 0.0001
            self.extend_len = 20 # Can be 20 also
            
            # Get the pickle file data
            self.pickle_file_path = data_path + f"{data_type}__data_collection.pickle"
            # Load the pickle file
            with open(self.pickle_file_path, "rb") as f:
                dataset = pickle.load(f)
            # Print the number of data
            print("Number of data: ", len(dataset))
            # Divide the dataset into train, validation, and test
            np.random.shuffle(dataset)
            train_val_test_ratio = [0.8, 0.1, 0.1]
            tvt_size = [int(len(dataset)*i) for i in train_val_test_ratio]
            train_data = dataset[:tvt_size[0]]
            val_data = dataset[tvt_size[0]:tvt_size[0]+tvt_size[1]]
            test_data = dataset[tvt_size[0]+tvt_size[1]:]
            # Print the number of data
            print("Number of train data: ", len(train_data))
            print("Number of validation data: ", len(val_data))
            print("Number of test data: ", len(test_data))
            
            
            train_data = mapsDataset(train_data, self.device)
            val_data = mapsDataset(val_data, self.device)
            test_data = mapsDataset(test_data, self.device)
            # Print the number of data
            print("Using dataset class:")
            print("Number of train data batch: ", len(train_data))
            
            # Make dataloaders
            self.trainDataloader = DataLoader(train_data, 
                                                batch_size=self.batch_size, 
                                                shuffle=True,
                                                collate_fn=train_data.rollout,
                                                drop_last=True)
            self.valDataloader = DataLoader(val_data, 
                                            batch_size=self.batch_size, 
                                            shuffle=True,
                                            collate_fn=val_data.rollout,
                                            drop_last=True)
            self.testDataloader = DataLoader(test_data, 
                                            batch_size=1, 
                                            shuffle=True,
                                            collate_fn=test_data.rollout,
                                            drop_last=True)
            
            print("Number of training dataloader: ", len(self.trainDataloader))
            print("Number of validation dataloader: ", len(self.valDataloader))
            print("Number of test dataloader: ", len(self.testDataloader))
            

            # Going to implement a Conditional Variational Autoencoder
            # The model will have an encoder and a decoder part
            # Encoder will take the input data and features and output in the latent space (mu and log_var)
            # Decoder will take the latent space and features and output the reconstructed data
            # Latent Space defintion: It is the space where the data is represented in a lower dimension
            # Defining model  
            
        elif test:
            dataset_dir = "./../motion_planning_datasets/"
            data_type = env_name
            data_path = dataset_dir + data_type + "/"
            self.test_dir = data_path + "test/"
            test_results = "./test_results/"
            if not os.path.exists(data_path):
                os.makedirs(data_path, exist_ok=True)
            self.img_dir = os.path.join(test_results, env_name + "__" + self.time_now)
            os.makedirs(self.img_dir, exist_ok=True)
        
        else:
            raise ValueError("Invalid mode")
        
        
        def make_model():
            # model parameters
            hidden_size = 4 # conditional_feature_dimension
            hidden_size_env1 = 256
            hidden_size_env2 = 128
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=3)
            self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=2)
            self.conv3 = nn.Conv2d(in_channels=5, out_channels=7, kernel_size=3, stride=1)
            self.env_fc1 = nn.Linear(6300, hidden_size_env1)
            self.env_fc2 = nn.Linear(hidden_size_env1, hidden_size_env2)
            self.env_fc3 = nn.Linear(hidden_size_env2, hidden_size)
            self.relu = nn.ReLU()

            x_dim = 2
            c_dim = hidden_size
            enc_h_dim1 = 256
            enc_h_dim2 = 128
            self.z_dim = 2 # latent_dimension - change to 4/8 and see the results for more complex learning
            # encoder part
            self.fc1 = nn.Linear(x_dim + c_dim, enc_h_dim1)
            self.fc2 = nn.Linear(enc_h_dim1, enc_h_dim2)
            self.fc31 = nn.Linear(enc_h_dim2, self.z_dim)
            self.fc32 = nn.Linear(enc_h_dim2, self.z_dim)

            dec_h_dim1 = 128
            dec_h_dim2 = 256
            dec_h_dim3 = 512
            # decoder part
            self.fc4 = nn.Linear(self.z_dim + c_dim, dec_h_dim1)
            self.fc5 = nn.Linear(dec_h_dim1, dec_h_dim2)
            self.fc6 = nn.Linear(dec_h_dim2, dec_h_dim3)
            self.fc7 = nn.Linear(dec_h_dim3, x_dim)

            self.optimizer = torch.optim.Adam(self.parameters(), \
                            lr=self.learning_rate, \
                            betas=[0.9, 0.999])
            self.mse_loss = torch.nn.MSELoss()
            self.kld_weight = 0.01
        
        make_model()
        
        
    def loss_function(self, recon_x, x, mu, log_var):
        MSE = self.mse_loss(recon_x, x)
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + self.kld_weight * KLD, MSE, KLD
    
    
    def env_encoder(self, env_batch):
        x = self.relu(self.conv1(env_batch))
        x = self.relu(self.conv2(x))
        feature_maps = self.relu(self.conv3(x))
        feature_maps = feature_maps.view(-1, feature_maps.size(1) * feature_maps.size(2) * feature_maps.size(3))
        x = self.relu(self.env_fc1(feature_maps))
        x = self.relu(self.env_fc2(x))
        x = self.relu(self.env_fc3(x))
        return x
    
    def cvae_decoder(self, z, cond_features):
        concat_input = torch.cat([z, cond_features], 1) 
        h = F.relu(self.fc4(concat_input))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        return self.fc7(h)
    
    
    def forward(self, env, label):
        # Encode the environment features
        env_feature = self.env_encoder(env)
        cond_features = env_feature

        # Encode the input label and conditional features
        concat_input = torch.cat([label, cond_features], 1)
        h = F.relu(self.fc1(concat_input))
        h = F.relu(self.fc2(h))
        mu = self.fc31(h)
        log_var = self.fc32(h)

        # Sample from the latent space
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add(mu)

        # Decode the latent representation and conditional features
        concat_input = torch.cat([z, cond_features], 1)
        h = F.relu(self.fc4(concat_input))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        recon_x = self.fc7(h)

        return recon_x, mu, log_var

    def hybrid_sampler(self, 
                       planning_env, 
                       start_point = (3, 3),
                       end_point = (96, 96),
                       _lambda=0.5, 
                       num_samples=1):
        samples = []
        for _ in range(num_samples):
            if random.uniform(0, 1) > _lambda:
                random_sample = (random.uniform(self.x_lims[0], self.x_lims[1] - 1), 
                                random.uniform(self.y_lims[0], self.y_lims[1] - 1)) if random.uniform(0, 1) > 0.05 else end_point
            else:
                with torch.no_grad():
                    z = torch.randn(1, self.z_dim).to(self.device)
                    env_feature = self.env_encoder(torch.from_numpy(np.array([[planning_env.get_env_image()]])).float().to(self.device))
                    conditional_features = env_feature
                    sample = self.cvae_decoder(z, conditional_features) 
                    random_sample = (sample[0].cpu().numpy()[0], 
                                     sample[0].cpu().numpy()[1])
                    random_sample = (random_sample[0] * 100, random_sample[1] * 100) # Assuming the image size is 100x100
            samples.append(random_sample)
        return samples

    def fit(self):
        print("Start training...")
        
        # Print basic information about training
        print(f"Training Data: {len(self.trainDataloader)} batches")
        print(f"Validation Data: {len(self.valDataloader)} batches")
        print(f"Test Data: {len(self.testDataloader)} batches")
        print(f"Training Workspaces: {len(self.train_workspace)}")
        print(f"Validation Workspaces: {len(self.val_workspace)}")
        print(f"Test Workspaces: {len(self.test_workspace)}")
        print(f"Number of Epochs: {self.num_epochs}")
        
        start_time = time.time()  # Track total training time
        train_step = 0  # Count gradient steps
        
        # Training Loop
        for epoch in range(self.num_epochs):
            self.train()  # Set model to training mode
            epoch_start_time = time.time()  # Measure epoch time
            
            train_loss, train_mse, train_kld = self._run_epoch(self.trainDataloader, train_step)
            val_loss, val_mse, val_kld = self.evaluate(epoch, start_point=(3,3), goal_point=(96,96))
            
            # Logging and performance metrics
            print(f"Epoch {epoch:03d} | Train Loss [Total: {train_loss:.5f} | MSE: {train_mse:.5f} | KLD: {train_kld:.5f}] "
                f"| Val Loss [Total: {val_loss:.5f} | MSE: {val_mse:.5f} | KLD: {val_kld:.5f}] "
                f"| Time: {time.time() - epoch_start_time:.2f}s")
            
            # Save the model periodically
            if epoch % 10 == 0:
                model_path = f"{self.checkpoint_dir}/epoch_{epoch:03d}-{val_loss:.5f}.pt"
                checkpoint = {
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                torch.save(checkpoint, model_path)
                print(f"Model saved to {model_path}")
        
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    # Sub-functions for modularity
    def _run_epoch(self, dataloader, train_step):
        """Run a single epoch."""
        total_loss, total_mse, total_kld = 0, 0, 0
        num_batches = len(dataloader)
        
        for env_batch, label_batch in dataloader:
            recon_x_batch, mu, log_var = self.forward(env_batch, label_batch)
            loss, mse_loss, kld_loss = self.loss_function(recon_x_batch, label_batch, mu, log_var)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_kld += kld_loss.item()
            train_step += 1
        
        # Compute average metrics
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_kld = total_kld / num_batches
        return avg_loss, avg_mse, avg_kld

    def evaluate(self,
                epoch,
                start_point=(3,3),
                goal_point=(96,96)):
        """Compute validation loss for training."""
        self.eval()
        loss_sum = 0
        mse_loss_sum = 0
        kld_loss_sum = 0
        count = 0

        with torch.no_grad():
            # Loop through the validation dataset
            for data_batch in self.valDataloader:  
                count += 1
                env_batch, label_batch = data_batch
                
                # Forward pass
                recon_x_batch, mu, log_var = self.forward(env_batch, label_batch)
                
                # Compute loss
                loss, mse_loss, kld_loss = self.loss_function(recon_x_batch, label_batch, mu, log_var)
                
                # Accumulate losses
                loss_sum += loss.item()
                mse_loss_sum += mse_loss.item()
                kld_loss_sum += kld_loss.item()

        # Return average losses over all batches
        val_loss = loss_sum / count
        val_mse = mse_loss_sum / count
        val_kld = kld_loss_sum / count
        return val_loss, val_mse, val_kld


    # Make a Kernel Density Estimation (KDE) plot
    def make_KDE(self, env, start_point, goal_point, _lambda=0.5, num_samples=1):
        # Get the random samples and then use the samples to get the KDE
        samples = self.hybrid_sampler(env, start_point, goal_point, _lambda, num_samples)
        x_pos, y_pos = np.array(samples)[:, 0], np.array(samples)[:, 1]
        x_pos, y_pos = np.array(x_pos), np.array(y_pos)
        xx, yy = np.meshgrid(np.linspace(0, 100, 100), np.linspace(0, 100, 100))
        xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
        xy_train = np.vstack([y_pos, x_pos]).T
        kde_skl = KDE2D(bandwidth=3.0)
        kde_skl.fit(xy_train)
        zz = np.reshape(np.exp(kde_skl.score_samples(xy_sample)), xx.shape)
        return xx, yy, zz


    def test(self, 
            model_path, 
            start_point=(3,3), 
            goal_point=(96,96), 
            env_name='forest',
            num_test_samples=5):
        """Test a trained-model"""
        self.eval()

        
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        

        success_rate = 0
        # Check if test_dir exists
        if not os.path.exists(self.test_dir):
            raise ValueError("Test directory does not exist")
        test_images = os.listdir(self.test_dir)
        print("Number of test images: ", len(test_images))
        
        # If num_test_samples -> 0, then test all the images
        # If num_test_samples !=0 and < len(test_images), then test num_test_samples
        # If num_test_samples > len(test_images), then test all the images after printing a warning
        if num_test_samples == 0:
            print("Testing all the images")
        elif num_test_samples < len(test_images):
            print(f"Testing {num_test_samples} images")
            test_images = test_images[:num_test_samples]
        else:
            print(f"Warning: num_test_samples is greater than the number of test images, testing all {len(test_images)} images instead")
            
        for idx, img in tqdm(enumerate(test_images)):
            image_path = os.path.abspath(self.test_dir + img)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Resize to ensure the image is 100x100
            image = cv2.resize(image, (100, 100))
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            
            env = Env2D()
            env.initialize(image_path, self.env_params)
            SMP = RRTstar(start_point, 
                          env, 
                          extend_len=self.extend_len)
            
            
            total_iteration = 0
            path_cost = 0
            final_path = None
            test_imgs = list()
            image_iterations = []
            total_iterations_to_find_path = 500
            
            for k in range(total_iterations_to_find_path):  
                env.initialize_plot(start_point, goal_point, plot_grid=False)
                env.visualize_environment()
                # Select one such sample based on model-random on a 50-50 basis
                selected_sample = self.hybrid_sampler(env, start_point, goal_point, _lambda=0.5, num_samples=1)
                selected_sample = selected_sample[0]

                xx, yy, zz = self.make_KDE(env, start_point, goal_point, _lambda=0.90, num_samples=100)
                env.plot_pcolor(xx, yy, zz, alpha=0.80)
                env.plot_tree(SMP.get_rrt(), color= '#00FFFF', linewidth=1.5)
                env.plot_state(selected_sample, color='black')
                env.plot_title(f"Iteration: {k}")
                env.visualize_environment()
                image_iterations.append(mplfig_to_npimage(env.figure))
                # plt.close()
                
                
                q_new = SMP.extend(selected_sample)
                if not q_new:
                    continue
                
                SMP.rewire(q_new)
                if SMP.is_goal_reached(q_new, goal_point):
                    SMP._q_goal_set.append(q_new)
                    total_iteration = k
                    final_path = SMP.reconstruct_path(q_new)
                    path_cost = SMP.cost(q_new)
                    break
                
            # If the path is found, plot the path from start to goal
            success_current = 1 if final_path != None else 0
            success_rate += success_current
            if final_path != None:
                env.initialize_plot(start_point, goal_point, plot_grid=False)
                env.plot_title(f"Past found in {total_iteration} iterations")
                env.plot_tree(SMP.get_rrt(),'#00FFFF', linewidth=1.5)
                env.plot_path(final_path)
                data = mplfig_to_npimage(env.figure)  # convert it to a numpy array
                image_iterations.append(data)
                print("Total solution cost: ", path_cost)
            else:
                print("Solution not found, moving to the next image but saving the images")
            name = f"{self.img_dir}/{env_name}_{idx}_iterations_{total_iteration}"
            make_video(image_iterations, name + '.mp4')
            
        print("Success rate: ", success_rate / len(test_images))