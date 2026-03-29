#!/usr/bin/env bash

if [ $# -ne 1 ]; then
  exit 1
fi

N="$1"
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -le 0 ]; then
  exit 1
fi

numProc=1
while [ "$numProc" -lt "$N" ]; do
  numProc=$((numProc * 2))
done

mpic++ -O2 -Wall -Wextra -pedantic -std=c++17 -o mes mes.cpp

dd if=/dev/random bs=1 count="$N" of=numbers 2>/dev/null

mpirun -np "$numProc" ./mes

rm -f mes numbers