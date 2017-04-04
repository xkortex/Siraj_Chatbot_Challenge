# Siraj_Chatbot_Challenge
Entry for machine learning 


### Dependencies
keras - obviously
h5py - for model checkpointing
keras-tqdm - because my Jupyter notebooks freezes on the default Keras progbar. Also, it's awesome

## Running 
To run the command line interface, just type:
`python main.py` 

### Arguments
`python main.py -c 2` to switch to double supporting facts dataset
`python main.py -m modelname.hdf5` to specify a custom model name. Note that the software automatically places these in the folders `models/c1/` or `models/c2/` depending on the dataset.


If you want to suppress some of the TF notifications and the progbars, you can append ` 2> /dev/null` to redirect that junk. 

You'll be able to train the network directly from the menu. If you do not have any trained models yet, you can select `f` from the menu to fit the model. 
