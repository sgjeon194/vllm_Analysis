# Test for LoRA with different batch size

i=1
while [ $i -le 32 ]
do
    echo "Running with batch_size=$i"
    sudo -E /usr/local/cuda/bin/nsys profile -o batch_size_${i}_lin_256 \
        --gpu-metrics-device=0 --cpuctxsw=none --force-overwrite true \
        --trace=cuda,nvtx \
        --cuda-graph-trace=node \
        .venv/bin/python examples/multilora_inference.py --batch_size $i --lin 256

    i=$(( i*2 ))
done    