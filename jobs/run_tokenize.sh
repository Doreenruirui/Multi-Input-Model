source /home/rui/.bash_profile
cd /home/rui/Project/Multi-Input-Model
python prepare_data.py \
        --data_dir=/home/rui/Dataset/OCR/$1 \
        --prefix=$2 
