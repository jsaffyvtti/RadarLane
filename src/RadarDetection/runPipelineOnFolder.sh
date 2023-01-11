#!/bin/bash

for file in *.bag;
    do python3 runPipeline.py -f "$file"; 
 done;