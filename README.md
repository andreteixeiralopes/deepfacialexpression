# deepfacialexpression

Introduction

This repository contains all information, data and source-code used in my Master Thesys.

Pre-Requisites
-> CUDA
-> OpenCv
-> Caffe

------------------------------------- CUDA INSTALLATION (for Linux Ubuntu 12.04 LTS) -------------------------------------

1. Check for CUDA device on the computer. If any, the description of the devices will be shown.
	lspci | grep -i nvidia

2. Check for the GCC installation
  gcc --version

3. Download the CUDA version for your Graphics Card architecture and Operational System. Here I downloaded version 7.0
  https://developer.nvidia.com/cuda-toolkit-70

4. Add the downloaded file to the linux repository
  sudo dpkg -i cuda-repo-<distro>_<version>_<architecture>.deb

5. Update the repository
  sudo apt-get update

6. Install
  sudo apt-get install cuda

7. Update Environment Variables
  export PATH=/usr/local/cuda-7.0/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH

8. Test
  nvcc -V

9. (Optional) Install examples.
  /usr/loca/cuda-7.0/cuda-install-samples-7.0.sh /home

------------------------------------- END OF CUDA INSTALLATION -------------------------------------


---------------------------------------- CAFFE INSTALLATION ----------------------------------------

1. General Dependences
  sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
  sudo apt-get install --no-install-recommends libboost-all-dev

2. BLAS
  sudo apt-get install libatlas-base-dev

3. OpenCv
  http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html

4.  Other Dependences (for Ubuntu 12.04 LTS)
  -> glog
	wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
	tar zxvf glog-0.3.3.tar.gz
	cd glog-0.3.3
	make && make install

  -> gflags
	wget https://github.com/schuhschuh/gflags/archive/master.zip
	unzip master.zip
	cd gflags-master
	mkdir build && cd build
	export CXXFLAGS="-fPIC" && cmake .. && make VERBOSE=1
	make && make install

	-> lmdb
	git clone https://github.com/LMDB/lmdb
	cd lmdb/libraries/liblmdb
	make && make install
	
5. Download Caffe last version from
  https://github.com/BVLC/caffe

6. Goto the caffe source folder and type
  cp Makefile.config.example Makefile.config

7. Compile
  cmake .
  make all
