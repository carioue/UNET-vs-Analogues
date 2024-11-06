# UNET-vs-Analogues

Codes associated with Estimation of the atmospheric circulation contribution to the European temperature variability with convolutional neural network, submitted to GRL. Associated data is available here: ??

### Analogues
- ANALOGUES1_find_analogues.py : this script calculates the analogues of each day of the target member among the other 40 members.
- ANALOGUES2_compare40members.py : it compares the analogues calculated on the 40 members (with ANALOGUES1_find_analogues.py script) and select the 20 smallests for each day.
- ANALOGUES3_temperature_reconstruction.py : it reconstructs daily temperature anomalies from the analogues selected with the previous script.

### UNET
- UNET1_training_40members.py : this script contains the structure of the UNET (the CTL experiment) and its training.
- UNET2_test_UNET.py : this script test the UNET trained with the previous script and save the results on the .npy file.
- UNET3_training_CNN_year_season_day.py : this script corresponds to the structures dans the trainings of the CNNs used to predict seasons, years and days.
- model_40members_input3286 : this is the UNET trained to predict European temperatures from the SLP of the large domain (CTL experiment). It can be load with the keras command load_model('model_40members_input3286').
- model_day : model trained to predict the day
- model_season : model trained to predict the season
- model_year : model trained to predict the year

### Plot figures
- Figures.ipynb : in the paper the figures correspond the 10 reconstructions. Here this notebook has been adapted to plot the results for only one reconstruction and the data associated to this notebook is available on ??
