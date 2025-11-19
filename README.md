# ECE570TinyReproductionProject

### Experment Setup
1. Run init_data.py to download and setup the dataset.
   ```bash
   python init_data.py
   ```
   Note: To maintain the TinyReproduction spirit of this project, this script only keeps the 'zara1' folder from the original dataset. You can modify the `folder_to_keep` variable in `init_data.py` to download more or different data as needed. The results of the experiments were gathered using only the 'zara1' data so they may vary depending on the data downloaded.
2. Install required packages.
   ```bash
   pip install -r requirements.txt
    ```

### Viewing the Data
1. Run the vanilla data visualization script.
   ```bash
   python vanilla_data_loader.py
    ```
2. Run the group data visualization script.
   ```bash
   python sg_data_loader.py
   ```

### Running Experiments
1. Run the Vanilla-LSTM train script.
   ```bash
   python train_vanilla_lstm.py
   ```
2. Run the Social-LSTM train script.
   ```bash
   python train_sg_lstm.py
    ```
Note: You can modify hyperparameters such as learning rate, batch size, and number of epochs directly in the training scripts before running them.
   
### Results
Results will be saved under the plots/ and plots_sg_lstm/ directories that are created after running the training scripts.

  