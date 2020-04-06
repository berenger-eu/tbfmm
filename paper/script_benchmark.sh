#!/bin/bash

function execute(){
    for nb_particles_height in "1000000:5" "10000000:6" ; do
        nb_particles=$(echo $nb_particles_height | cut -d':' -f 1)
        height=$(echo $nb_particles_height | cut -d':' -f 2)
        
        echo "nb_paticles $nb_paticles"
        echo "height $height"
        
        for nb_threads in 1 2 4 8 16 ; do
            echo "$nb_threads"
            OMP_PROC_BIND=true OMP_NUM_THREADS=$nb_threads ./testUnifKernel -nb $nb_particles -th $height -nc
        done
    done
}

execute;

