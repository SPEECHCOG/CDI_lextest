## Word separability test for words in CDI words and gestures (short form; North American English).


## Getting started

* Run your model on the audio files at `CDI_synth/` and extract embeddings as corresponding .txt files into `/path_to/extracted/embeddings/`. Embeddings can be one embedding per .wav or frame-level embeddings.

One embedding per wav: each .txt file should have one vector on the first row, float values separated by white spaces.
Frame-level embeddings per wav: each .txt file should have one frame per row, embedding values as floats separated by white spaces.  

* Run evaluation software to get overall separability score.


## Running from command line (tested on Narvi and Puhti clusters)

1) Get a CPU node.  

2) Load MATLAB if not present. e.g.:  

`module load matlab`  

3a) For utterance-level embeddings, execute:  

`sh CDI_lextest.sh '/path_to/original/audios/' '/path_to/extracted/utt_level_embeddings/`

3b) For frame-level embeddings, execute:  

`sh CDI_lextest.sh '/path_to/original/audios/' '/path_to/extracted/frame_level_embeddings/' 'full' 0`

or for parallel computing (recommended if parfor available):  

`sh CDI_lextest.sh '/path_to/original/audios/' '/path_to/extracted/frame_level_embeddings/' 'full' 1`

4) Results will be written to `output.txt`in CDI_lextest main folder. If you want to specify different output folder for the results, give the path as the fifth argument, e.g.:  

`sh CDI_lextest.sh '/path_to/original/audios/' '/path_to/extracted/frame_level_embeddings/' 'full' 1 /path_to/output/folder/`

By default, audio files are located in `CDI_synth/` of this repository.



## Running from MATLAB desktop

You can run the code as a normal MATLAB script by calling CDI_lextest.m directly (the same syntax as above).

## Baseline replication with provided log-Mel example embeddings

In order to replicate baselines with log-Mel features, run:

`sh CDI_lextest.sh 'CDI_synth/' 'demodata/CDI_embs_uttlevel/'`

or

`sh CDI_lextest.sh 'CDI_synth/' 'demodata/CDI_embs_frame/' 'full' 1`

which should produce 17.10% and 47.08%, respectively.

## Citing

If you use this test or its derivations in publications, please cite the following paper:

Khorrami, K., Cruz Blandón, M. A., & Räsänen, O. (2023). Computational insights to acquisition of phonemes, words, and word meanings in early language: sequential or parallel acquisition? Proc. CogSci-2023, Sydney, Australia.
