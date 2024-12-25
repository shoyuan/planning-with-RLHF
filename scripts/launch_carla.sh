#!/bin/bash

for (( i=1; i<=$1; i++ ))
do
    port=$((i*$2))
    fuser $port/tcp -k
    fuser $((port + 1))/tcp -k
    fuser $((port + 2))/tcp -k
    $HOME/CARLA_0.9.10.1/CarlaUE4.sh -world-port=$port -vulkan -world-port=$port &
done
wait
