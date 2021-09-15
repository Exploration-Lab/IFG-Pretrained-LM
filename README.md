# IFG-Pretrained-LM

Pre-trained Language Models as Prior Knowledge for Playing Text-based Games [[arxiv](https://arxiv.org/abs/2107.08408)]

Set up:
```bash
conda create -n {yourenvname} python=3.7 anaconda
pip install torch==1.4 transformers==2.5.1 jericho fasttext wandb importlib_metadata
python -m spacy download en_core_web_sm
git clone repo
```

Get the trained distillBERT [here](https://drive.google.com/drive/folders/1-2w-SwDzSUlEEgKL62jxLotuPUiTRTE4?usp=sharing).
OR Run LM training from ```dbert_train``` folder.

Run RL training:
```bash
conda activate {yourenvname}
cd repo/dbert_drrn
python train.py --rom_path ../games/{gamefilename} --lm_path {lm_path} --output_dir ./logs
```
Get more game engines [here](https://github.com/BYU-PCCL/z-machine-games/tree/master/jericho-game-suite).

## Acknowledgements
The code borrows from [CALM](https://github.com/princeton-nlp/calm-textgame) and [huggingface](https://huggingface.co/).

## Citation 
If you want to use our work in your research, please cite:
```
@misc{singh2021pretrained,
      title={Pre-trained Language Models as Prior Knowledge for Playing Text-based Games}, 
      author={Ishika Singh and Gargi Singh and Ashutosh Modi},
      year={2021},
      eprint={2107.08408},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

