# interactive_fiction_games

Set up:
```bash
conda create -n {yourenvname} python=3.7 anaconda
pip install torch==1.4 transformers==2.5.1 jericho fasttext wandb importlib_metadata
python -m spacy download en_core_web_sm
git clone repo
```

Get the trained distillBERT [here](https://drive.google.com/drive/folders/1-2w-SwDzSUlEEgKL62jxLotuPUiTRTE4?usp=sharing).
OR
Run LM training:
```bash
--
```

Run RL training:
```bash
conda activate {yourenvname}
cd repo/dbert_drrn
python train.py --rom_path ../games/{gamefilename} --lm_path {lm_path} --output_dir ./logs
```
Get more game engines [here](https://github.com/BYU-PCCL/z-machine-games/tree/master/jericho-game-suite).

## Acknowledgements
The code borrows from [CALM](https://github.com/princeton-nlp/calm-textgame).