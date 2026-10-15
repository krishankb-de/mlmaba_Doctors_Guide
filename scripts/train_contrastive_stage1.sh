#!/bin/bash
#SBATCH --partition=@SLURM_PARTITION@
#SBATCH --account=@SLURM_ACCOUNT@
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --job-name=simcse_stage1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail

echo "=== JOB START (Stage 1: SimCSE on PubMed, LM-pretrained init) ==="
date
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo ""
echo "Configuration:"
echo "  - Init: outputs/hybrid_70m_lm_pretrain_pubmed/checkpoints/last.ckpt"
echo "  - Dataset: PubMed abstracts (ccdv/pubmed-summarization)"
echo "  - Mode: SimCSE (text-only contrastive)"
echo "  - Batch size: 32 (effective 64 w/ grad accum 2)"
echo "  - Sequence length: 512 (matches Stage 0)"
echo "  - Learning rate: 2e-5"
echo "  - Warmup steps: 500"
echo "  - Max steps: 10,000"
echo "  - Grad clip: 1.0"
echo ""

cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm"

export HF_HOME="$PWD/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING=0

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment .venv not found!"
    exit 1
fi
source .venv/bin/activate

if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not found in virtual environment!"
    exit 1
fi

echo ""
echo "Verifying CUDA availability:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"
echo ""

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA is not available to PyTorch!"
    exit 1
fi

nvidia-smi

LM_CKPT="./outputs/hybrid_70m_lm_pretrain_pubmed/checkpoints/last.ckpt"
if [ ! -f "$LM_CKPT" ]; then
    echo "ERROR: LM-pretrained checkpoint not found: $LM_CKPT"
    echo "Run scripts/train_stage0_lm_pubmed.sh first."
    exit 1
fi

python scripts/train_contrastive.py \
  --config-name config_70m \
  dataset=pubmed \
  trainer=a100_single_gpu \
  trainer.accelerator=cuda \
  trainer.max_epochs=-1 \
  trainer.max_steps=10000 \
  trainer.accumulate_grad_batches=2 \
  trainer.val_check_interval=500 \
  trainer.log_every_n_steps=25 \
  contrastive_mode=simcse \
  dataset.batch_size=32 \
  dataset.eval_batch_size=32 \
  dataset.max_length=512 \
  dataset.max_seq_length=512 \
  dataset.num_workers=4 \
  dataset.preprocessing_num_workers=4 \
  dataset.pin_memory=true \
  lm_checkpoint="$LM_CKPT" \
  callbacks.checkpoint.every_n_train_steps=1000 \
  callbacks.checkpoint.save_top_k=3 \
  callbacks.early_stopping.enabled=true \
  callbacks.early_stopping.monitor=train/contrastive_loss_epoch \
  callbacks.early_stopping.patience=3 \
  callbacks.early_stopping.min_delta=0.01 \
  experiment_name=stage1_pubmed_simcse \
  output_dir=./outputs/stage1_pubmed_simcse \
  wandb.enabled=false \
  model.learning_rate=2.0e-5 \
  model.warmup_steps=500 \
  model.gradient_clip_val=1.0

echo ""
echo "=== JOB END (Stage 1: SimCSE) ==="
echo "Checkpoint saved to: ./outputs/stage1_pubmed_simcse/checkpoints/"
date
