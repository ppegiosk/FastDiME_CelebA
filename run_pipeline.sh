MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 500 --learn_sigma True --noise_schedule linear --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 10 --timestep_respacing 200"
DATAPATH=/scratch/ppar/data/img_align_celeba
MODELPATH=/scratch/ppar/models/ddpm-celeba.pt
CLASSIFIERPATH=/scratch/ppar/models/classifier.pth
ORACLEPATH=/scratch/ppar/models/oracle.pth
OUTPUT_PATH=/scratch/ppar/results/
NUMBATCHES=99999999
METHOD=fastdime
EXPNAME=fastdime_shortcut
C=1
c=0

# parameters of the dataset
QUERYLABEL=31 # 31: 'smile'
TARGETLABEL=-1
IMAGESIZE=128

# parameters of the sampling
SEED=4
GPU=0
USE_LOGITS=True
CLASS_SCALES='8,10,15'
LAYER=18
PERC=30
L1=0.05
S=60

# parameters of the self-optimized masking
DILATION=5
THRES=0.15
WARMUP=30

# generate shortcut counterfactuals using FastDiME
mpiexec -n 1 python -W ignore main.py --method $METHOD $MODEL_FLAGS $SAMPLE_FLAGS \
  --dataset 'ShortcutCelebA' --image_size $IMAGESIZE --num_batches $NUMBATCHES \
  --query_label $QUERYLABEL --target_label $TARGETLABEL \
  --output_path $OUTPUT_PATH --exp_name $EXPNAME --gpu $GPU --seed $SEED  \
  --model_path $MODELPATH --classifier_path $CLASSIFIERPATH --oracle_path $ORACLEPATH \
  --start_step $S --classifier_scales $CLASS_SCALES --l_perc $PERC --l_perc_layer $LAYER --l1_loss $L1 \
  --dilation $DILATION --masking_threshold $THRES --warmup_step $WARMUP \
  --save_images True --save_x_t False --save_z_t False \
  --use_sampling_on_x_t True --use_logits $USE_LOGITS \
  --num_chunks $C --chunk $c \

# run shortcut learning detection pipeline
python shortcut_detection.py --percentage 1.0
python shortcut_detection.py --percentage 0.75
python shortcut_detection.py --percentage 0.5