#!/bin/bash
if (($# == 0))
then
  gpu_id=0
else
  gpu_id=$1
fi
echo "use GPU: $gpu_id"

min_memory_usage=5000
min_gpu_usage=65
max_wait_time=30
idle_time=0

while true 
do
  gpu_usage=`nvidia-smi -i ${gpu_id} --query-gpu=utilization.gpu --format=csv,noheader,nounits`

  # check idle state
  if ((${gpu_usage}<${min_gpu_usage}))
  then
    ((idle_time++))
    echo "idle: ${idle_time}/${max_wait_time}"
  else
    idle_time=0
  fi

  # if idle reach limit, start process
  if ((${idle_time}>=${max_wait_time}))
  then
    echo "max idle time reach, restart process"
    # kill old process
    for line in $(seq 16 30)
    do
      process_memory=`nvidia-smi -i ${gpu_id} | awk -v a=$line '$8 ~/[0-9]+/{if((NR==a)) {print $8}}'`
      # check process memory is large enough
      if ((${#process_memory}>3))
      then
        if ((${process_memory:0:-3}>${min_memory_usage}))
        then
          process_id=`nvidia-smi -i ${gpu_id} | awk -v a=$line '$5 ~/[0-9]+/{if((NR==a)) {print $5}}'`
          echo 'ready to kill ' $process_id $process_memory $gpu_usage '%'
          kill -9 $process_id
        fi
      fi
    done
    tmux send-keys -t ":0.0" "echo start new run" Enter
    tmux send-keys -t ":0.0" "cd /home/pcy/rl/srl; conda activate rlgpu; export LD_LIBRARY_PATH=/home/pcy/miniconda3/envs/rlgpu/lib:$LD_LIBRARY_PATH" Enter
    tmux send-keys -t ":0.0" "python run.py +corl=28ho name=28ho random_seed=$2 gpu_id=$1" Enter
    idle_time=0
  fi
  sleep 1
done