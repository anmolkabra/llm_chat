#!/usr/bin/env bash

remote_cluster=$1
compute_node=$2
port=8501
ssh -t -t $remote_cluster -L $port:localhost:$port ssh $compute_node -L $port:localhost:$port
