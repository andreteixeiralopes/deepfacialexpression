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
	 
#Get the Source
1. Download the **caffe** and **data** directories from this repository. After the download, the directories showld be in the same parent directory.

2. Go to caffe directory

3. Compile
  	- cmake .
  	- make all

4. Everything showld compile without errors.

#Get the Data
1. Request your copy of the Cohn-Kanade database from
	- http://www.consortium.ri.cmu.edu/ckagree/

2. The database needs to be separated in the eigth non-overlap groups, to perform the experiments in the right way. To separate the data, extract the Cohn-Kanade data to the folders G1 to G8, put the files in theses folders according to the file **label.txt**.

3. To replicate the experiments described in the dissertation the sinthetic samples need to be generated. To perform this, use the **generateData.cpp** code, stored in the tools folder.

4. To generate the synthetic data, from the caffe root directory, run:
	- make all
	- ./tools/generateData

5. Open the file data/synthetic/solver.prototxt and change the firts line, the path to the file **train.prototxt** should contains the absolute path to the file (the file is in the same folder as the solver.prototxt).

#Run Training
1. The training source-code is stored in the file **trainDeepFace.cpp**, inside the tools folder.

2. From the caffe root directory, run:
	- make all
	- ./tools/trainDeepFace

#Run Testing

2. Open the file **trainDeepFace.cpp** and change the method called in tha Main, to test(). Remember of commenting the line that calls the train() method.

3. From the caffe root directory, run:
	- make all
	- ./tools/trainDeepFace

4. Evaluate: The files with the patter **summary_GT*IT0*.** are the best results, selected with the validation group. The **.txt** files, contains the confusion matrixes and the accuracy for both classifiers, the n-class and the binary. The **.net** files are the networks weights that archieve the results shown in the text files.
