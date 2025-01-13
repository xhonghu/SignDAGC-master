#!/bin/bash
nohup python main.py > /dev/null 2>&1 &
sleep 30
nohup python /root/data1/demo.py > /dev/null 2>&1 &
nvitop