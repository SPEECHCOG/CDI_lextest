## Word separability test for words in CDI words and gestures (short form; North American English).


#### Getting started

* Run your model on the audio files and extract embeddings as corresponding .txt files into `/path_to/extracted/embeddings/`. Embeddings can be one embedding per .wav or frame-level embeddings.

One embedding per wav: each .txt file should have one vector on the first row, float values separated by white spaces.
Frame-level embeddings per wav: each .txt file should have one frame per row, embedding values as floats separated by white spaces.  

* Run evaluation software to get overall separability score.


#### Running from command line (tested on Narvi)

Get a CPU node.

`module load matlab`  


`matlab -batch 'CDI_lextest '/path_to/original/audios/' '/path_to/extracted/embeddings/';' > output.txt`  



`matlab -batch 'CDI_lextest '/path_to/original/audios/' '/path_to/extracted/embeddings/' 'full' 1;' > output.txt`  

A cleaner output.txt version looks like this:  

`matlab -batch 'CDI_lextest '/path_to/original/audios/' '/path_to/extracted/embeddings/' 'single' 0;' > tmp.txt; grep "recall" tmp.txt | { grep -v grep || true; } > output.txt;rm tmp.txt`
