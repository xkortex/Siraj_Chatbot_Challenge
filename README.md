# Siraj_Chatbot_Challenge
Entry for machine learning 


### Dependencies
- pandas - Needed for some data sorting operations
- keras - obviously
- h5py - for model checkpointing
- keras-tqdm - because my Jupyter notebooks freezes on the default Keras progbar. Also, it's awesome

## Running 
To run the command line interface, just type:
`python main.py` 

### Arguments
- `-m {modelname}` - Set the name of the model and weight save file
- `-c {N}` - Run challenge mode N. `1` is Single context bAbI, `2` is Double context bAbI
- `-v` - Verbose flag

Example uses:
<br>`python main.py -c 2` to switch to double supporting facts dataset
<br>`python main.py -m modelname.hdf5` to specify a custom model name. Note that the software automatically places these in the folders `models/c1/` or `models/c2/` depending on the dataset.





If you want to suppress some of the TF notifications and the progbars, you can append ` 2> /dev/null` to redirect that junk. 

You'll be able to train the network directly from the menu. If you do not have any trained models yet, you can select `f` from the menu to fit the model. 

# Network Improvements:

Here are some improvements I made to the demo network:
### Bidirectional LSTM
The single forward-pass LSTM was converted to bidirectional layer with the Bidirectional wrapper. Yuuuuuge improvement on double-context task - 84.7% after 260 epochs with single, improved to 90% after only 110 epochs with bidirectional. Nice! Asymptoted to 95% after about 150 epochs. The Single-context task got to 90% after 60 vs 85 epochs, modest improvement.  

These are all accuracy numbers, not validation accuracy. Valacc is still falling a bit behind, so there is a fair amount of overfit going on. 

## Not-so-improvements
### Third LSTM layer
Adding a forward pass after the bidirectional pair did not give improvements, in fact it caused the network to stall out around 55%. I've seen towers of LSTMs used to good effect in other NLP papers. Maybe they have some secret sauce I don't. 
