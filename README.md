# Variational Autoencoder (VAE) with Spatial Statistics Loss
A PyTorch implementation of generative models to generate novel chemical materials.

To run training,
1. Download multiple_lines.zip dataset at https://drive.google.com/file/d/1QpHOx9C_nBF6hrKfws5Q6hPdKKTzYXJG/view?usp=sharing.
2. Unzip multiple_lines.zip and place the multiple_lines inside the data folder in the root directory.
3. Open multiple_lines and unzip images.zip.
4. Run setup.sh by running ./setup.sh in terminal. If you're running the training for the first time, the code will
ask for a WandB API key. Enter a valid API key. This command should run the src/models/sweep.py file which contains the configuration parameters used in our tinypaper. To see our tiny paper, see toward_learning_latent_variable.pdf.

Quantitative Results of Original VAE and Our VAE
We conducted an additional experiment, training our Variational Auto-encoder (VAE) on a more complex dataset. Initially, our VAE was trained on simple datasets consisting of randomly generated horizontal and vertical lines, with each sample containing one to five lines. For our latest experiment, we enhanced the dataset complexity: each sample now includes three to seven lines. Despite this change, we kept the training configuration for both the "vanilla" VAE and our VAE unchanged.

In this experiment, we noted a significant increase in the mean data space Mean Squared Error (MSE) for our model, rising by 68%, and the mean spatial statistics MSE increased by 168%. The "vanilla" VAE also showed an increase in the mean data space MSE by 102% and in the mean spatial statistics MSE by 300%. Interestingly, while the mean data space MSE of our model was previously 3.60 times that of the “vanilla” VAE, it reduced to 3.02 times in the recent experiment. Similarly, the mean spatial stats MSE of our model, initially 0.50 times that of the “vanilla” VAE, decreased to 0.27 times in this experiment.

These results (see ./harder_dataset_results) indicate that both models' performance declined when faced with a more challenging dataset. However, it's evident that our model outperformed the "vanilla" VAE. This difference in performance can be attributed to our model's superior efficiency in learning the data distribution, especially in contrast to the “vanilla” VAE, which struggles more due to the increased complexity and higher number of pixels to recall.

