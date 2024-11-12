#!/usr/bin/env bash

compute_node=$1
port=8501
ssh -t -t aida -L $port:localhost:$port ssh $compute_node -L $port:localhost:$port
