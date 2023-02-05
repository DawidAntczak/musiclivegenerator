
# musiclivegenerator
Tools created while working on the master's thesis: ***Automatic music generation methods for video games***

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


The repository contains the model, second and third (last) step of data preparation, scripts to run WebSocket server for live generation and script to generate some sample MIDI files.

The trained model is included in *save* directory as *everything-game-30s-transposed.sess*.

## Generation
- To generate MIDI file samples with various conditioning:
    ```shell
    Run generate.py 
    ```
- To start WebSocket server for live generation:
    ```shell
    Run generate_live.py 
    ```


## Training

- [First data preparation step](https://github.com/DawidAntczak/musiclivegeneratorcsharp)

- Second data preparation step:
    ```shell
    Run data_prep.py 
    ```

- Preprocessing:
    ```shell
    Run preprocess.py 
    ```

- Training
    ```shell
    Run train.py
    
## Requirements
Everything was run using Python 3.8 on Windows in Anaconda environment exported to [requirements.txt](https://github.com/DawidAntczak/musiclivegenerator/blob/main/requirements.txt).


## Acknowledgement
Code based on implementation of [EmotionBox](https://github.com/KaitongZheng/EmotionBox), the paper is available on 
[https://arxiv.org/abs/2112.08561](https://arxiv.org/abs/2112.08561).

[Original license file included](https://github.com/DawidAntczak/musiclivegenerator/blob/main/ORIGINAL_LICENSE)
