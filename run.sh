#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="llama_adapter"
MODEL_PATH="/data_new/private/tuyuge"
VERSION="7b"
DATASET="Alpaca"

OPTS=""
OPTS+=" --dataset ${DATASET}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${MODEL_PATH}/results/llama-${VERSION}"
OPTS+=" --save ${MODEL_PATH}/results"
OPTS+=" --save-name finetune-llama-${DATASET}"

#  tweak hyperparameters:
OPTS+=" --batch-size 16"
OPTS+=" --epochs 2"
OPTS+=" --train-iters 1000"
OPTS+=" --save-iters 1000"
OPTS+=" --max-length 512"
OPTS+=" --lr 1e-3"
OPTS+=" --inspect-iters 1000"
OPTS+=" --warmup-iters 100"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1.0e-2" #noam initial lr: 0.01/math.sqrt(100)*1/100=1e-05
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --start-step 0"

# run training and save to log:
TODAY=$(date +%Y-%m-%d)
CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} $BASE_PATH/finetune_llama.py ${OPTS}"
echo ${CMD}
${CMD} 2>&1 | tee ${BASE_PATH}/logs/finetune-llama-${VERSION}-${DATASET}-${TODAY}.log