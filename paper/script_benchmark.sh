#!/bin/bash

function execute(){
    all_lines="nb_particles,height,nb_threads,exec_time,build_time,algo_name"
    
    for nb_particles_height in "1000000:5" "10000000:6" ; do
        nb_particles=$(echo $nb_particles_height | cut -d':' -f 1)
        height=$(echo $nb_particles_height | cut -d':' -f 2)
        
        echo "nb_particles $nb_particles"
        echo "height $height"
        
        for nb_threads in 1 2 4 8 16 32 ; do
            echo "$nb_threads"
            result=$(OMP_PROC_BIND=true OMP_NUM_THREADS=$nb_threads ./testUnifKernel -nb $nb_particles -th $height -nc)
            exec_time=$(echo "$result" | grep "Execute FMM in" | cut -d' ' -f 4 | cut -d's' -f 1)
            build_time=$(echo "$result" | grep "Build the tree in" | cut -d' ' -f 5 | cut -d's' -f 1)
            algo_name=$(echo "$result" | grep "Algorithm name" | cut -d' ' -f 3)
                        
            line="$nb_particles,$height,$nb_threads,$exec_time,$build_time,$algo_name"
            echo "$line"
            all_lines="$all_lines\n$line"
        done
    done
    
    echo -e "$all_lines" > "timings.csv"
}

execute;

