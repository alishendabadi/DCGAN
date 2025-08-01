# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
./install.sh
```
This will install all required packages and run tests to verify everything works.

### 2. Quick Test (Recommended First)
```bash
python3 quick_test.py
```
This runs training for just 5 epochs to verify everything works correctly.

### 3. Full Training
```bash
python3 train_dcgan.py
```
This runs the complete training for 100 epochs.

## 📁 What You'll Get

After training, you'll have:
- `samples/` - Generated images during training
- `checkpoints/` - Saved model states
- `data/` - MNIST dataset (downloaded automatically)

## 🎨 Generate Samples

```bash
# Generate 16 random samples
python3 generate_samples.py --model_path checkpoints/final_model.pth

# Generate interpolation between two digits
python3 generate_samples.py --model_path checkpoints/final_model.pth --interpolation
```

## ⚙️ Customize Training

Edit `config.py` to modify:
- Number of epochs
- Batch size
- Learning rate
- Model architecture

## 🐛 Troubleshooting

- **CUDA out of memory**: Reduce batch size in `config.py`
- **Import errors**: Run `./install.sh` again
- **Poor results**: Increase number of epochs

## 📊 Expected Results

- **5 epochs**: Basic digit-like shapes
- **25 epochs**: Recognizable digits
- **100 epochs**: High-quality handwritten digits

## 🎯 Tips

1. Use GPU if available for faster training
2. Monitor loss values during training
3. Check generated samples every 10 epochs
4. Be patient - GANs need time to converge 