#!/bin/bash

# unset http proxy which maybe set by docker daemon
export http_proxy=""; export https_proxy=""; export no_proxy=""; export HTTP_PROXY=""; export HTTPS_PROXY=""; export NO_PROXY=""

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/

PY=python3
if [[ -z "$WS" || $WS -lt 1 ]]; then
  # WS=并发数量
  WS=1
fi

# 无限循环执行任务,$1是函数参数，代表进程的编号 0\1\2\3
function task_exe(){
    while [ 1 -eq 1 ];do
      $PY rag/svr/task_executor.py $1;
    done
}

# 启动多个进程，在后台执行rag/svr/task_executor.py i
for ((i=0;i<WS;i++))
do
  task_exe  $i &
done

while [ 1 -eq 1 ];do
    $PY api/ragflow_server.py
done

wait;
