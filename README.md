# Siraj_Chatbot_Challenge
Entry for machine learning tutorial [How to Make a Chatbot - Intro to Deep Learning #12](https://www.youtube.com/watch?v=t5qgjJIBy9g)


### Dependencies
- pandas - Needed for some data sorting operations
- keras - Deep learning FTW
- tensorflow - My preferred Keras backend. Tensorboard is now integrated, so it'll need some tweaking to be Theano-compatible again 
- h5py - for model checkpointing and saving weights
- keras-tqdm - because my Jupyter notebooks freezes on the default Keras progbar. Also, it's awesome.

## Running 
To run the command line interface, just type:
`python main.py` 
If this is the first time you are running the program, or you just created a new model, you'll have to train it first, which you can do directly from the menu. If you do not have any trained models yet, you can select `f` from the menu to fit the model. 

### Arguments
- `-m {modelname}` - Set the name of the model and weight save file
- `-c {N}` - Run challenge mode N. `1` is Single context bAbI, `2` is Double context bAbI
- `-a {N}`- Run architecture N. `1` is bAbI DMN, `2` a more conventional convolutional LSTM (warning: hard on memory)
- `-v` - Verbose flag

Example uses:
<br>`python main.py -c 2` to switch to double supporting facts dataset
<br>`python main.py -m modelname.hdf5` to specify a custom model name. Note that the software automatically places these in the folders `models/c1/` or `models/c2/` depending on the dataset.


##### If you want to suppress some of the TF notifications and the progbars, you can append ` 2> /dev/null` to redirect that junk.

### Note on Challenge 2: Two Supporting Facts
There are actually two challenges that came with the Q/A task, the single supporting fact, and the double supporting facts. The former is pretty easy to knock out of the park, while the latter has proven quite stubborn. I was able to get >95% training accuracy but only 35-40% validation accuracy, a surefire sign of overfitting. I tried some clever hacks with the network but I was not able to improve results. The authors claim that they aced the two supporting fact problem, but the Keras code as provided seems to fall short. Meh.

# Network Improvements:

Here are some improvements I made to the demo network:
### Convo LSTM
I added the option to compare against a convolutional LSTM architecture. So far, kind of middling results. Needs to be configured for minibatch. 


### Bidirectional LSTM
The single forward-pass LSTM was converted to bidirectional layer with the Bidirectional wrapper. Yuuuuuge improvement on double-context task - 84.7% (training acc) after 260 epochs with single, improved to 90% after only 110 epochs with bidirectional. Nice! Asymptoted to 95% after about 150 epochs. However, I later realized these figures were pretty misleading, as the validation was not keeping pace with the training accuracy.

 The Single-context task got to 90% validation accuracy after 60 vs 85 epochs, modest improvement.

### Time-Distributed Dense
Adding a TDD layer before the LSTM gave an additional jump in terms of training time and overall accuracy, reaching 95% valacc after 65 epochs on single-context (with default 32 nodes).

These are all accuracy numbers, not validation accuracy. Valacc is still falling a bit behind, so there is a fair amount of overfit going on. 

## Not-so-improvements
### Conv1D in the Match layer
Who doesn't love convo layers? Hoping to get better context recognition, I put a convolayer after the Match dot product part of the network. It didn't hurt the performance, but it didn't give the gains in the Challenge 2 I was looking for.

### Third LSTM layer
Adding a forward pass after the bidirectional pair did not give improvements, in fact it caused the network to stall out around 55%. I've seen towers of LSTMs used to good effect in other NLP papers. Maybe they have some secret sauce I don't. 


## Navigating
The main menu looks like this:
```bash
------------------------------
Sandra went back to the kitchen.
Sandra journeyed to the garden.
Mary went back to the kitchen.
Sandra went to the kitchen.
------------------------------

..: Back
 1: Load Random Story
 2: Query
 3: Query (loop)
 f: Fit for N epochs
 q: Quit
Enter menu selection: 
```
The currently loaded story is shown at the top. Enter letters to navigate the menu. 
- `..` is currently non-functional
- `1` loads a new story. 
- `2` lets you type in a query. It goes back to the main menu after
- `3` like 2, but brings you back to the query prompt after, for convenience
- `f` lets you enter in a number to fit the model for that many epochs. 

