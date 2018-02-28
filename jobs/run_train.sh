cd /home/rui/Project/Multi-Input-Model
python -m pdb train.py \
    --data_dir='/home/rui/Dataset/OCR/'$1 \
    --train_dir='/home/rui/Model/OCR/nlc/'$2 \
    --num_layers=$3 \
    --size=$4 \
    --gpu_frac=1.0 \
    --print_every=200 
