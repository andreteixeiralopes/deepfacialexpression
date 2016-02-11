# deepfacialexpression

#Introduction
This repository contains all information, data and source-code used in my Master Thesys.


#Pre-Requisites

- CUDA

- OpenCv

- Caffe


#### CUDA INSTALLATION (for Linux Ubuntu 12.04 LTS)

1. Check for CUDA device on the computer. If any, the description of the devices will be shown.
	- lspci | grep -i nvidia

2. Check for the GCC installation
	- gcc --version

3. Download the CUDA version for your Graphics Card architecture and Operational System.
	- https://developer.nvidia.com/cuda-toolkit-70

4. Add the downloaded file to the linux repository
	- sudo dpkg -i cuda-repo-<distro>_<version>_<architecture>.deb

5. Update the repository
	- sudo apt-get update

6. Install
	- sudo apt-get install cuda

7. Update Environment Variables
	- export PATH=/usr/local/cuda-7.0/bin:$PATH
	- export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH

8. Test
	- nvcc -V

9. (Optional) Install examples.
	- /usr/loca/cuda-7.0/cuda-install-samples-7.0.sh /home


#### CAFFE INSTALLATION

1. General Dependences
	- sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
	- sudo apt-get install --no-install-recommends libboost-all-dev

2. BLAS
	- sudo apt-get install libatlas-base-dev

3. OpenCv
	- http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html

4.  Other Dependences (for Ubuntu 12.04 LTS)
	- wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
	- tar zxvf glog-0.3.3.tar.gz
	- cd glog-0.3.3
	- make && make install

	- wget https://github.com/schuhschuh/gflags/archive/master.zip
	- unzip master.zip
	- cd gflags-master
	- mkdir build && cd build
	- export CXXFLAGS="-fPIC" && cmake .. && make VERBOSE=1
	- make && make install

	- git clone https://github.com/LMDB/lmdb
	- cd lmdb/libraries/liblmdb
	- make && make install
	 
5. Download Caffe last version from
  	- https://github.com/BVLC/caffe

6. Goto the caffe source folder and type
  	- cp Makefile.config.example Makefile.config

7. Compile
  	- cmake .
  	- make all

#Get the Source
1. Download the souce files **util.h** and **utilCaffe.h** and put in the Caffe include folder

2. Download the source files **trainDeepFace.cpp** and **generateData.cpp** and put it in the Caffe tools folder.

4. Go to Caffe directory

5. Compile
  	- cmake .
  	- make all

#Get the Data
1. Request your copy of the Cohn-Kanade database from
	- http://www.consortium.ri.cmu.edu/ckagree/

2. The database needs to be separated in the eigth non-overlap groups, to perform the experiments in the right way. To separate the data, extract the Cohn-Kanade data and create the folders G1 to G8, and put the files in theses folders according to the file **label-original.txt**.

3. To replicate the experiments described in the dissertation the sinthetic samples need to be generated. To perform thism, use the **generateData.cpp** code. In the begining of the file, change the values of the variables **originalDataFolder** and **syntheticDataFolder**, to the folder that contains the orignal data separated in the groups, with the **label-original.txt** file, and to the folder where the synthetic data will be stored (the groups folder, G1 to G8, need to create before the execution of this method).

4. After these operations, run:
	- make all
	- ./generateData

