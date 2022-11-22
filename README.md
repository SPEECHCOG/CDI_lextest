## Word separability test for words in CDI words and gestures (short form; North American English).


#### Getting started

* Run your model on the audio files and extract embeddings as corresponding .txt files into `/path_to/extracted/embeddings/`. Embeddings can be one embedding per .wav or frame-level embeddings.

One embedding per wav: each .txt file should have one vector on the first row, float values separated by white spaces.
Frame-level embeddings per wav: each .txt file should have one frame per row, embedding values as floats separated by white spaces.  

* Run evaluation software to get overall separability score.


#### Running from command line (tested on Narvi)

1) Get a CPU node.  

2) Load MATLAB if not present. e.g.:  

`module load matlab`  

3a) For utterance-level embeddings, execute:  

`sh CDI_lextest.sh '/path_to/original/audios/' '/path_to/extracted/utt_level_embeddings/`

3b) For frame-level embeddings, execute:  

`sh CDI_lextest.sh '/path_to/original/audios/' '/path_to/extracted/frame_level_embeddings/' 'full' 0`

or for parallel computing (recommended if parfor available):  

`sh CDI_lextest.sh '/path_to/original/audios/' '/path_to/extracted/frame_level_embeddings/' 'full' 1`

4) Results will be written in output.txt    
