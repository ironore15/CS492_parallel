#/bin/bash

g++ sparse.cpp mmreader.cpp -o sparse -lpthread -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include

./sparse ./matrix/hood.mtx 256
./sparse ./matrix/2cubes_sphere.mtx 2048
./sparse ./matrix/cage12.mtx 1024
./sparse ./matrix/consph.mtx 2048
./sparse ./matrix/cop20k_A.mtx 2048
./sparse ./matrix/filter3D.mtx 2048
./sparse ./matrix/hood.mtx 1024
./sparse ./matrix/m133-b3.mtx 1024
./sparse ./matrix/mac_econ_fwd500.mtx 1024
./sparse ./matrix/majorbasis.mtx 1024
./sparse ./matrix/mario002.mtx 512
./sparse ./matrix/mc2depi.mtx 512
./sparse ./matrix/offshore.mtx 1024
./sparse ./matrix/patents_main.mtx 1024
./sparse ./matrix/pdb1HYS.mtx 4096
./sparse ./matrix/poisson3Da.mtx 16384
./sparse ./matrix/pwtk.mtx 1024
./sparse ./matrix/rma10.mtx 4096
./sparse ./matrix/scircuit.mtx 1024
./sparse ./matrix/shipsec1.mtx 1024
./sparse ./matrix/webbase-1M.mtx 256
