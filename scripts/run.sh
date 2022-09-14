#!/bin/bash
if (($# == 0))
then
  gpu_id=0
else
  gpu_id=$1
fi
echo "use GPU: $gpu_id"

seed=$2
echo "seed: $seed"
load_tag=$3
echo "load_tag: $load_tag"

if (($# == 3))
then
  monitor_gpu_id=$1
else
  monitor_gpu_id=$4
fi
echo "monitor gpu: $monitor_gpu_id"

min_memory_usage=5000
min_gpu_usage=65
max_wait_time=40
idle_time=0

while true 
do
  gpu_usage=`nvidia-smi -i ${monitor_gpu_id} --query-gpu=utilization.gpu --format=csv,noheader,nounits`

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
      process_memory=`nvidia-smi -i ${monitor_gpu_id} | awk -v a=$line '$8 ~/[0-9]+/{if((NR==a)) {print $8}}'`
      # check process memory is large enough
      if ((${#process_memory}>3))
      then
        if ((${process_memory:0:-3}>${min_memory_usage}))
        then
          process_id=`nvidia-smi -i ${monitor_gpu_id} | awk -v a=$line '$5 ~/[0-9]+/{if((NR==a)) {print $5}}'`
          echo 'ready to kill ' $process_id $process_memory $gpu_usage '%'
          kill -9 $process_id
        fi
      fi
    done
    tmux send-keys -t ":0.2" "echo start new run" Enter
    tmux send-keys -t ":0.2" "cd /home/pcy/rl/srl; conda activate rlgpu; export LD_LIBRARY_PATH=/home/pcy/miniconda3/envs/rlgpu/lib:$LD_LIBRARY_PATH" Enter
    tmux send-keys -t ":0.2" "python run.py +corl_rebuttal=26to name=26ho_$load_tag random_seed=$seed gpu_id=$1 load_tag=latest_$load_tag" Enter
    idle_time=0
  fi
  sleep 1
done
