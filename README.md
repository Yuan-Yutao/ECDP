# ECDP
Code for paper "Effcient Conditional Diffusion Model with Probability
Flow Sampling for Image Super-resolution".

## Required packages
```
pip install torch torchvision
pip install click lpips pillow piq pyyaml torchdiffeq tqdm
```

## Prepare datasets
Put datasets into data/..., for example, DIV2K dataset should be put in
`data/div2k/{DIV2K_train_HR,DIV2K_valid_HR,DIV2K_valid_LR_bicubic}`.
For more details, see code in `lib/datasets`.

## Training ECDP
First, train the RRDB encoder:
```
./main.py train --config conf/rrdbnet-df2k.yaml experiment-name
```

Then, extract the parameters of RRDB encoder from the checkpoint:
```python
s = torch.load('results/.../checkpoint.pt', map_location=torch.device('cpu'))
t = {}
for k, v in s['model'].items():
    if k.startswith('model.'):
        t[k[6:]] = v
torch.save(t, 'pretrained-rrdbnet-df2k.pt')
```

Train the ECDP model:
```
./main.py train --config conf/ecdp-df2k-train.yaml experiment-name
```

Finetune the ECDP model using image quality loss:
```
# You need to modify finetune_pretrained_model in the config file
./main.py train --config conf/ecdp-df2k-finetune.yaml experiment-name
```

## Testing ECDP
```
# Experiment name here should be those in results directory
./main.py test experiment-name --save-images
```

The images are now in the results directory.
