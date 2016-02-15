/*
 * trainDeepFace.cpp
 *
 *  Created on: Feb 11, 2016
 *      Author: alopes
 */

#include "util.h"
#include "utilCaffe.h"

/*
 * 19/11/2015 - No treinamento é escolhida a época que deu o melhor resultado. Por isso
 * a necessidade de separamos os dados tem três grupos, treino, validação e teste. O dado
 * de teste é utilizado somente para avaliar a rede, após todo o treinamento, que inclusive usa
 * a validação para determinar a melhor rede e época.
*/

int qtdeExpressoes = 6;
int qtdeGrupos = 8;
int qtdeIteracoes = 10;

string dataPath = "../data/synthetic/";
string defaultNetFile = "bstNetwork";

bool isOriginal(string fileName){
	return fileName.substr(fileName.size() - 7, 7) == "000.jpg";
}

int getClassCKJ(int i){
	//As the class 2 isn't in the database, we get all classes higher than 1 (3, 4, 5,6,7) and decreases, to get all classes in the interval 1-6
	if(i > 1){
		i--;
	}

	if(qtdeExpressoes == 6)
		return --i;
	else
		return i;
}

void readTrainData(TrainData* dadosTreino)
{
	char filename[100], charGrupo;
	int expressao, grupo;
	Mat fileImg;

	FILE *f = fopen((dataPath + "label.txt").c_str(), "r");

	while(!feof(f)){
		fscanf(f, "%c%d%s %d\n", &charGrupo, &grupo, filename, &expressao);
		if(expressao > 0 || qtdeExpressoes == 7){
			stringstream ss;
			ss << dataPath << "G" << grupo << filename;
			fileImg = imread(ss.str(),0);
			if(grupo == dadosTreino->testSet){
				if(isOriginal(string(filename))){
					dadosTreino->testSamples.push_back(fileImg);
					dadosTreino->testLabels.push_back(getClassCKJ(expressao));
					string individuo(filename);
					individuo = individuo.substr(1, 4);
					if(dadosTreino->IndividuosTeste.find(individuo) == dadosTreino->IndividuosTeste.end())
					{
						dadosTreino->IndividuosTeste[individuo] = 0;
					}
					dadosTreino->IndividuosTeste[individuo]++;
				}
			}
			else if(grupo == dadosTreino->validationSet){
				if(isOriginal(string(filename))){
					dadosTreino->validationSamples.push_back(fileImg);
					dadosTreino->validationLabels.push_back(getClassCKJ(expressao));
				}
			}
			else{
				if(fileImg.rows > 0){
					dadosTreino->trainSamples.push_back(fileImg);
					dadosTreino->trainLabels.push_back(getClassCKJ(expressao));
				}
			}
		}
	}
	fclose(f);

	while(dadosTreino->trainSamples.size() % 20 != 0){
		dadosTreino->trainSamples.erase(dadosTreino->trainSamples.begin());
		dadosTreino->trainLabels.erase(dadosTreino->trainLabels.begin());
	}
}

string gerarNomeArquivo(string resultsPath, int grupoTeste, int grupoValidacao, int iter, int individuo = -1){
	stringstream ss;
	ss << resultsPath;
	ss << "summary";
	ss << "_GT" << grupoTeste;
	ss << "_GV" << grupoValidacao;
	ss << "_IT" << iter;
	if( individuo >= 0){
		ss << "_IND" << individuo;
	}

	ss << ".txt";

	return ss.str();
}

void executarTeste(string pastaResultados, int grupoTeste, int grupoValidacao, int iteracao, vector<Mat> imagens, vector<int> labels, int individuo = -1, string defaultNetFile = "bstNetwork"){

	string nomeArquivoSaida = gerarNomeArquivo(pastaResultados, grupoTeste, grupoValidacao, iteracao, individuo);

	FILE *summaryLocal = fopen(nomeArquivoSaida.c_str(), "w");
	testNet(dataPath + "test.prototxt", defaultNetFile, imagens, labels, summaryLocal, dataPath + "solver.prototxt");

	nomeArquivoSaida = nomeArquivoSaida.substr(0, nomeArquivoSaida.size() - 3).append("net");

	copyFile(defaultNetFile, nomeArquivoSaida);

	fclose(summaryLocal);
}

void separarIndividuo(
		vector<Mat> Exemplos,
		vector<int> Labels,
		int individuo,
		vector<Mat>& ExemploSemIndividuo,
		vector<int>& LabelsSemIndividuo,
		vector<Mat>& ExemploIndividuo,
		vector<int>& LabelsIndividuo,
		map<string, int> Individuos)
{
	int atual = 0;
	for(int i = 0; i < Individuos.size(); i++)
	{
		if(i != individuo)
		{
			for(int j = 0; j < getValueAtIndex(i, Individuos);j++ ){
				ExemploSemIndividuo.push_back(Exemplos[atual]);
				LabelsSemIndividuo.push_back(Labels[atual]);
				atual++;
			}
		}
		else
		{
			for(int j = 0; j < getValueAtIndex(i, Individuos);j++ ){
				ExemploIndividuo.push_back(Exemplos[atual]);
				LabelsIndividuo.push_back(Labels[atual]);
				atual++;
			}
		}
	}
}

/*
	Experimento usando 7 grupos para treino, 1 grupo para validacao e teste. Este ultimo eh dividido
	em dois grupos, um com (n - 1) individuos para validacao e (1) individuo para teste.
*/
void train(){
	string solver = dataPath + "solver.prototxt";
	string resultsPath = dataPath + "results/";

	for(int grupoTeste = 1; grupoTeste <= qtdeGrupos; grupoTeste++){
		TrainData dadosTreino;
		dadosTreino.testSet = grupoTeste;
		//Garante que so vai separar em treino e teste, nao vai ter validacao
		dadosTreino.validationSet = 99;
		readTrainData(&dadosTreino);
		for(int individuo = 1; individuo <= dadosTreino.IndividuosTeste.size(); individuo++){
			double bstValidationAccuracy = 0, mediaNormal = 0, mediaBinario = 0;
			vector<Mat> imagensValidacao, imagensTeste;
			vector<int> labelsValidacao, labelsTeste;
			separarIndividuo(dadosTreino.testSamples, dadosTreino.testLabels,
					  individuo - 1, imagensValidacao,
					  labelsValidacao, imagensTeste,
					  labelsTeste, dadosTreino.IndividuosTeste);
			//Embaralha e trunca os dados de validacao
			shuffleVectors(imagensValidacao, labelsValidacao);
			while(imagensValidacao.size() % 55 != 0){
				imagensValidacao.erase(imagensValidacao.begin());
				labelsValidacao.erase(labelsValidacao.begin());
			}
			for(int iter = 1; iter <= qtdeIteracoes; iter++){
				shuffleVectors(dadosTreino.trainSamples, dadosTreino.trainLabels);
				trainNet(solver, dadosTreino.trainSamples, dadosTreino.trainLabels, imagensValidacao, labelsValidacao);

				executarTeste(resultsPath, grupoTeste, grupoTeste, iter, imagensTeste, labelsTeste, individuo);
			}
		}
	}
}

void test(string resultsPath = "results"){
	string solver = dataPath + "solver.prototxt";

	for(int grupoTeste = 1; grupoTeste <= 8; grupoTeste++){
		TrainData dadosTreino;
		dadosTreino.testSet = grupoTeste;
		//Garante que so vai separar em treino e teste, nai vai ter validacao
		dadosTreino.validationSet = 99;
		readTrainData(&dadosTreino);
		for(int individuo = 1; individuo <= dadosTreino.IndividuosTeste.size(); individuo++){
			double bstValidationAccuracy = 0;
			vector<Mat> imagensValidacao, imagensTeste;
			vector<int> labelsValidacao, labelsTeste;
			separarIndividuo(dadosTreino.testSamples, dadosTreino.testLabels,
					  individuo - 1, imagensValidacao,
					  labelsValidacao, imagensTeste,
					  labelsTeste, dadosTreino.IndividuosTeste);
			//Embaralha e trunca os dados de validacao
			shuffleVectors(imagensValidacao, labelsValidacao);
			for(int iter = 1; iter <= qtdeIteracoes; iter++){
				stringstream ss;
				ss << resultsPath << "summary_GT" << grupoTeste << "_GV" << grupoTeste << "_IT" << iter << "_IND" << individuo << ".net";
				cout << ss.str() << endl;
				double validationAccuracy = testNet(dataPath + "test.prototxt", ss.str(), imagensValidacao, labelsValidacao, NULL, solver);

				if(validationAccuracy > bstValidationAccuracy){
					bstValidationAccuracy = validationAccuracy;
					executarTeste(resultsPath, grupoTeste, grupoTeste, 0, imagensTeste, labelsTeste, individuo);
				}
			}
		}
	}
}

int main(){
	train();
	//test();
}



