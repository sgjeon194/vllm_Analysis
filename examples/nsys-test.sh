# Test for LoRA with different batch size

target_dir=".profiling_results"

if [ ! -d "$target_dir" ]; then
    mkdir "$target_dir"
    echo "Created directory: $target_dir"
fi

i=1
while [ $i -le 256 ]
do
    padded_i=$(printf "%03d" $i)
    echo "Running with batch_size=$i"
    sudo -E /usr/local/cuda/bin/nsys profile -o .profiling_results/batch_size_${padded_i}_lin_256_all_different \
        --gpu-metrics-device=0 --cpuctxsw=none --force-overwrite true \
        --trace=cuda,nvtx \
        --cuda-graph-trace=node \
        .venv/bin/python examples/multilora_inference.py --batch_size $i --lin 256

    i=$(( i*2 ))
    echo "============================= Finished!! ============================="
    echo ""
    echo ""
done    
