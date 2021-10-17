#!/bin/bash

# sbatch --partition=gpusmall --time=36:00:00 sbatch_train_mahti.sh
# sbatch --partition=gpumedium --time=36:00:00 sbatch_train_mahti.sh
# sbatch --partition=gputest --time=00:15:00 sbatch_train_mahti.sh

#SBATCH --job-name=s
#SBATCH --account=project_2000936
#SBATCH --chdir=/projappl/project_2000936/SpecVQGAN
#SBATCH --output=./sbatch_logs/run_%J.txt
#SBATCH --error=./sbatch_logs/run_%J.txt
#SBATCH --verbose
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:4,nvme:700
#SBATCH --mem=64G

# exit when any command fails
set -e
# if some error happens in the initialation of parallel process then you can get the debug info.
export NCCL_DEBUG=INFO  # comment it if you are not debugging distributed parallel setup
export NCCL_DEBUG_SUBSYS=ALL # comment it if you are not debugging distributed parallel setup

# We want names of master and slave nodes. Make sure this node (MASTER) comes first
MASTER=`/bin/hostname -s`
if (( $SLURM_JOB_NUM_NODES > 1 )); then
    SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
fi
# Get a random unused port on this host(MASTER)
MPORT=`comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
# it is ok if slaves are not defined
HOSTLIST="$MASTER $SLAVES"

# Determine the number of GPU
NUM_GPUS="${CUDA_VISIBLE_DEVICES//[^[:digit:]]/}"
NUM_GPUS=${#NUM_GPUS}

# Feature extraction commands
EXTRACT_SPECS_VGGSOUND="echo \"Extracting input features @ \$HOSTNAME\"
    for i in \`seq 1 64\`; do
        i=\`printf %02d \$i\` \
        && tar -xf $SCRATCH/vladimir/vggsound/mel_tars/melspec_10s_22050hz_\$i.tar -C $LOCAL_SCRATCH \
        &
    done; wait"
EXTRACT_FEATS_VGGSOUND="echo \"Extracting video features @ \$HOSTNAME\"
    for i in \`seq 1 64\`; do
        i=\`printf %02d \$i\` \
        && tar -xf $SCRATCH/vladimir/vggsound/feature_tars/feature_flow_bninception_dim1024_21.5fps_\$i.tar -C $LOCAL_SCRATCH \
        && tar -xf $SCRATCH/vladimir/vggsound/feature_tars/feature_rgb_bninception_dim1024_21.5fps_\$i.tar -C $LOCAL_SCRATCH \
        &
    done; wait"

EXTRACT_SPECS_VAS="echo \"Extracting input features\"
strings=(\"dog\" \"fireworks\" \"drum\" \"baby\" \"gun\" \"sneeze\" \"cough\" \"hammer\")
for class in \"\${strings[@]}\"; do
    mkdir -p $LOCAL_SCRATCH/features/\$class && \
    tar xf $SCRATCH/vladimir/vas/feature_tars/\${class}_melspec_10s_22050hz.tar -C $LOCAL_SCRATCH/features/\$class &
done; wait"
EXTRACT_FEATS_VAS="echo \"Extracting video features\"
strings=(\"dog\" \"fireworks\" \"drum\" \"baby\" \"gun\" \"sneeze\" \"cough\" \"hammer\")
for class in \"\${strings[@]}\"; do
    tar xf $SCRATCH/vladimir/vas/feature_tars/\${class}_feature_rgb_bninception_dim1024_21.5fps.tar -C $LOCAL_SCRATCH/features/\$class & \
    tar xf $SCRATCH/vladimir/vas/feature_tars/\${class}_feature_flow_bninception_dim1024_21.5fps.tar -C $LOCAL_SCRATCH/features/\$class &
done; wait"


# Path to experiment
EXPERIMENT_PATH=./logs/2021-06-09T15-17-18_vas_transformer

# Select a dataset here
DATASET="VAS"
# DATASET="VGGSound"

# TOP_K_OPTIONS=( "1" "16" "64" "100" "128" "256" "512" "1024" )
TOP_K_OPTIONS=( "64" )
# VGGSOUND_SAMPLES_PER_VIDEO=10
VGGSOUND_SAMPLES_PER_VIDEO=1
# VAS_SAMPLES_PER_VIDEO=100
VAS_SAMPLES_PER_VIDEO=10

if [[ "$DATASET" == "VGGSound" ]]; then
    # EXTRACT_FILES_CMD="$EXTRACT_SPECS_VGGSOUND"
    EXTRACT_FILES_CMD="$EXTRACT_SPECS_VGGSOUND && $EXTRACT_FEATS_VGGSOUND"
    SPEC_DIR_PATH="$LOCAL_SCRATCH/melspec_10s_22050hz/"
    RGB_FEATS_DIR_PATH="$LOCAL_SCRATCH/feature_rgb_bninception_dim1024_21.5fps/"
    FLOW_FEATS_DIR_PATH="$LOCAL_SCRATCH/feature_flow_bninception_dim1024_21.5fps/"
    SAMPLES_FOLDER="VGGSound_test"
    SPLITS="\"[test, ]\""
    SAMPLER_BATCHSIZE=32
    SAMPLES_PER_VIDEO=$VGGSOUND_SAMPLES_PER_VIDEO
elif [[ "$DATASET" == "VAS" ]]; then
    # EXTRACT_FILES_CMD="$EXTRACT_SPECS_VAS"
    EXTRACT_FILES_CMD="$EXTRACT_SPECS_VAS && $EXTRACT_FEATS_VAS"
    SPEC_DIR_PATH="$LOCAL_SCRATCH/features/*/melspec_10s_22050hz/"
    RGB_FEATS_DIR_PATH="$LOCAL_SCRATCH/features/*/feature_rgb_bninception_dim1024_21.5fps/"
    FLOW_FEATS_DIR_PATH="$LOCAL_SCRATCH/features/*/feature_flow_bninception_dim1024_21.5fps/"
    SAMPLES_FOLDER="VAS_validation"
    SPLITS="\"[validation, ]\""
    SAMPLER_BATCHSIZE=4
    SAMPLES_PER_VIDEO=$VAS_SAMPLES_PER_VIDEO
else
    echo "NotImplementedError"
    exit
fi

# Some info to print
echo "Local Scratch" $LOCAL_SCRATCH
echo "Hostlist:" $HOSTLIST
echo "Samples per video:" $SAMPLES_PER_VIDEO "; Sampler path" $EXPERIMENT_PATH
echo $EXTRACT_FILES_CMD
echo $SPEC_DIR_PATH
echo $RGB_FEATS_DIR_PATH
echo $FLOW_FEATS_DIR_PATH

# Extracting files at every node
for node in $HOSTLIST; do
    ssh -q $node "$EXTRACT_FILES_CMD" &
done
wait

for TOP_K in "${TOP_K_OPTIONS[@]}"; do
    echo "Starting TOP-$TOP_K. Number of Samples/Video: $SAMPLES_PER_VIDEO"

    # Saving the time stamp to reuse it when sampling is done. Random second is used to avoid overalapping \
    # sample folder names
    RAND_SEC=$((RANDOM%59+1))
    NOW=`date +"%Y-%m-%dT%H-%M-$(printf %02d $RAND_SEC)"`

    # Launch the torch.distributed.launch tool, first on master (first in $HOSTLIST) then on the slaves
    # Escape the '$' from the variable if you want to take variable from the server's environement (where you ssh)
    # By default bash will execute the tring with the variables from this script
    # loading conda environment.
    # We are doing both sampling and evaluation sequentially
    source $PROJAPPL/miniconda3/etc/profile.d/conda.sh
    NODE_RANK=0
    for node in $HOSTLIST; do
        ssh -q $node "conda activate specvqgan && \
                    cd $PROJAPPL/SpecVQGAN && \
                    python -m torch.distributed.launch \
                        --nproc_per_node=$NUM_GPUS \
                        --nnodes=$SLURM_JOB_NUM_NODES \
                        --node_rank=$NODE_RANK \
                        --master_addr=$MASTER \
                        --master_port=$MPORT \
                        --use_env \
                            evaluation/generate_samples.py \
                            sampler.config_sampler=evaluation/configs/sampler.yaml \
                            sampler.model_logdir=$EXPERIMENT_PATH \
                            sampler.splits=$SPLITS \
                            sampler.samples_per_video=$SAMPLES_PER_VIDEO \
                            sampler.batch_size=$SAMPLER_BATCHSIZE \
                            sampler.top_k=$TOP_K \
                            data.params.spec_dir_path=$SPEC_DIR_PATH \
                            data.params.rgb_feats_dir_path=$RGB_FEATS_DIR_PATH \
                            data.params.flow_feats_dir_path=$FLOW_FEATS_DIR_PATH \
                            sampler.now=$NOW && \
                    python -m torch.distributed.launch \
                        --nproc_per_node=$NUM_GPUS \
                        --nnodes=$SLURM_JOB_NUM_NODES \
                        --node_rank=$NODE_RANK \
                        --master_addr=$MASTER \
                        --master_port=$MPORT \
                        --use_env \
                        evaluate.py \
                            config=./evaluation/configs/eval_melception_${DATASET,,}.yaml \
                            input2.path_to_exp=$EXPERIMENT_PATH \
                            patch.specs_dir=$SPEC_DIR_PATH \
                            patch.spec_dir_path=$SPEC_DIR_PATH \
                            patch.rgb_feats_dir_path=$RGB_FEATS_DIR_PATH \
                            patch.flow_feats_dir_path=$FLOW_FEATS_DIR_PATH \
                            input1.params.root=$EXPERIMENT_PATH/samples_$NOW/$SAMPLES_FOLDER
                    " &
        NODE_RANK=$((NODE_RANK+1))
    done
    wait
    echo "Done TOP-$TOP_K"
done
