# AI-generated-chemical-materials
A PyTorch implementation of generative models to generate novel chemical materials.

To run training,
1. Download multiple_lines.zip dataset at https://drive.google.com/file/d/1QpHOx9C_nBF6hrKfws5Q6hPdKKTzYXJG/view?usp=sharing.
2. Unzip multiple_lines.zip and place the multiple_lines inside the data folder in the root directory.
3. Open multiple_lines and unzip images.zip.
4. Run setup.sh by running ./setup.sh in terminal. If you're running the training for the first time, the code will
ask for a WandB API key. Enter a valid API key. This command should run the src/models/sweep.py file which contains the configuration parameters used in our tinypaper. To see our tiny paper, see toward_learning_latent_variable.pdf.

Quantitative Results of Original Variational Auto-encoder (VAE) and Our VAE
We train a “vanilla” VAE that minimizes the distance in data space between the original and the reconstruction (“data space loss”), and our proposed model, which minimizes the distance in spatial statistics space between the original and the reconstruction (“spatial statistics loss”). Our method produces reconstructions that are closer together in spatial statistics space, but further apart in data space, indicating more diverse samples. This is to be expected: we directly try to make the reconstructions be as close as possible as the original in spatial statistics space, and do not try to match the original in data space at all. Note the exact reconstructions that are close to the original in data space would also be close to the original in spatial statistics space (see ./results/validation_reconstructions_combined_histograms.pdf). For that reason, the spatial statistics error when training with data space loss is larger, but not much larger, than when training with data space loss.
To view example reconstructions from both the vanilla VAE and our modified VAE, please refer to ./results/reconstructions. It's important to note that the Mean Square Error (MSE) values listed in these files are organized in ascending order. Additionally, for comparison purposes, we have included the reconstructions from the alternate model in each respective file.