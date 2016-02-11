/*
 * utilCaffe.h
 *
 *  Created on: 02/07/2015
 *      Author: alopes
 */

#include "util.h"

void trainNet(string solverFile, vector<Mat> trainSamples, vector<int> trainLabels, vector<Mat> testSamples, vector<int> testLabels)
{
	//Set which device will be used
	Caffe::SetDevice(0);

	//Set the GPU processing
	Caffe::set_mode(Caffe::GPU);

	//Object that will store the network parameters
	SolverParameter solverParamters;

	//Read the network parameters
	ReadProtoFromTextFileOrDie(solverFile, &solverParamters);

	//Creates the solver
	SGDSolver<float> solver(solverParamters);

    shared_ptr<MemoryDataLayer<float> > train_data_layer;
    train_data_layer = boost::dynamic_pointer_cast<MemoryDataLayer<float> >(solver.net()->layers()[0]);
    train_data_layer->AddMatVector(trainSamples, trainLabels);

    shared_ptr<MemoryDataLayer<float> > test_data_layer;
    test_data_layer = boost::dynamic_pointer_cast<MemoryDataLayer<float> >(solver.test_nets()[0]->layers()[0]);
    test_data_layer->AddMatVector(testSamples, testLabels);

    //Start the training procedure
    solver.Solve();
}

double accuracyNormal = -1;
double accuracyBinaria = -1;

/* This function evaluate the test data in a trained network
 * Parameters:
 * 	prototxtFile - File containing the network structure and layers
 * 	caffeModelFile - File containing the weights, result of the training step
 * Output: Print's the accuracy of the network over all samples and the confidence result of each sample
*/
double testNet(string caffeLayerDefinition, string caffeTrainedLayers, vector<Mat> testSamples, vector<int> testLabels, FILE* summaryFile, string caffeSolverFile)
{
	//Set which device will be used
	Caffe::SetDevice(0);

	//Set the GPU processing
	Caffe::set_mode(Caffe::GPU);

	//Create the test network based on the prototxt file
	Net<float> caffe_net(caffeLayerDefinition, caffe::TEST);


	if(summaryFile){
		//Object that will store the network parameters
		SolverParameter solverParamters;

		//Read the network parameters
		ReadProtoFromTextFileOrDie(caffeSolverFile, &solverParamters);

		fprintf(summaryFile,"Network Parameters:\n");
		fprintf(summaryFile,"Learning Rate:\t\t%f\n", solverParamters.base_lr());
		fprintf(summaryFile,"Momentum:\t\t\t%f\n", solverParamters.momentum());
		fprintf(summaryFile,"Max Iterations:\t\t%d\n", solverParamters.max_iter());
		fprintf(summaryFile,"Weight Decay:\t\t%f\n", solverParamters.weight_decay());
		fprintf(summaryFile,"Gamma:\t\t\t\t%f\n", solverParamters.gamma());
		fprintf(summaryFile,"Power:\t\t\t\t%f\n\n", solverParamters.power());

	}

	//Load the weights to the network
	caffe_net.CopyTrainedLayersFrom(caffeTrainedLayers);


	//The data layer (first layer) of this network is a memory data layer, so we need to
	//input its values
	shared_ptr < MemoryDataLayer<float > > dataLayer =
			dynamic_pointer_cast < MemoryDataLayer<float > > (caffe_net.layers()[0]);

	//Add the data to the network
	dataLayer->AddMatVector(testSamples, testLabels);

	//The amount of the hits
	double hits = 0;
	float loss = 0;
	int nClasses = 6;
	int confusionMatrix[nClasses][nClasses];
	int acertosBinario[nClasses];
	int totalBinario[nClasses];
	for(int i =0; i < nClasses; i++){
		for(int j = 0; j < nClasses; j++){
			confusionMatrix[i][j] = 0;
		}
		acertosBinario[i] = 0;
		totalBinario[i] = 0;
	}

	for(int k = 0; k < testSamples.size(); k++){
		//Recognize the current sample
		
		//clock_t ini = clock();
		caffe_net.ForwardPrefilled(&loss);
		//clock_t fim= clock();
		//cout << float(fim - ini) / CLOCKS_PER_SEC << endl;
		//getchar();

		//Get the data of the last layer after the evaluation
		shared_ptr < Blob<float > > prob = caffe_net.blob_by_name("ip2");

		float maxval = 0;
		int maxinx = 0;

		//Get the maximum confidence level of the classes
		for (int i = 0; i < prob->count(); i++) {
			float val = prob->cpu_data()[i];
			if (val > maxval) {
				maxval = val;
				maxinx = i;
				}
		}

		//Hit
		if(maxinx == testLabels[k]){
			hits++;
			for(int i = 0; i < nClasses; i++){
				acertosBinario[i]++;
				totalBinario[i]++;
			}
		}
		else
		{
			for(int i = 0; i < nClasses; i++){
				if(i != testLabels[k] && i != maxinx){
					//acertosBinario[testLabels[k]]++;
					acertosBinario[i]++;
				}
				//totalBinario[testLabels[k]]++;
				totalBinario[i]++;
			}
		}

		confusionMatrix[testLabels[k]][maxinx]++;

	}

	double accuracy = hits / testSamples.size();

	if(summaryFile){
		fprintf(summaryFile, "Confusion Matriz\n");
		for(int i =0; i < nClasses; i++){
			for(int j = 0; j < nClasses; j++){
				fprintf(summaryFile,"%d\t", confusionMatrix[i][j]);
			}
			fprintf(summaryFile, "\n");
		}

		fprintf(summaryFile, "\nAccuracy by class\n");
		for(int i =0; i < nClasses; i++){
			int classHit = 0;
			float classTotal = 0;
			for(int j = 0; j < nClasses; j++){
				classTotal += confusionMatrix[i][j];
			}
			fprintf(summaryFile, "Class %d: %d/%.0f = %.2f%\n", (i+1), confusionMatrix[i][i], classTotal, (confusionMatrix[i][i]/ classTotal)*100.0);
		}

		fprintf(summaryFile,"\n\nTotal Accuracy: %.0f/%d = %.2f%", hits, testSamples.size(), accuracy*100.0);

		fprintf(summaryFile,"\n\n\n\nTeste Binário\n");
		int _acertos = 0;
		int _total = 0;
		for(int i =0; i < nClasses; i++){
			fprintf(summaryFile, "Class %d: %d/%d = %.2f%\n", (i+1), acertosBinario[i], totalBinario[i], (acertosBinario[i] / (totalBinario[i]*1.0))*100.0);
			_acertos += acertosBinario[i];
			_total += totalBinario[i];
		}

		fprintf(summaryFile,"\n\nTotal Accuracy: %.d/%d = %.2f%", _acertos, _total, (_acertos / (_total*1.0))*100.0);
		accuracyNormal = accuracy * 100.0;
		accuracyBinaria=(_acertos / (_total*1.0))*100.0;
	}


	return accuracy;
}


