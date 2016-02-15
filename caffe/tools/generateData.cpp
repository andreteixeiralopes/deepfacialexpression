#include <cstring>
#include <map>
#include <string>
#include <string.h>
#include <vector>
#include "caffe/caffe.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <fstream>
#include <boost/smart_ptr.hpp>
#include "glog/logging.h"
#include <time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sstream>
#include <queue>
#include <boost/filesystem.hpp>
#include "util.h"
#include "time.h"

using namespace std;
using namespace cv;
using namespace caffe;
using namespace boost;
using namespace boost::filesystem3;

#define YXMin 1.3
#define YXMax 3.2
#define XXMin 1.2
#define XXMax 1.2

RNG rng;

string originalDataFolder = "../../data/";
string syntheticDataFolder = "../../data/synthetic/";

typedef
		struct
		Sample
		{
			string samplePath;
			string fileName;
			int group;
			int label;
			Sample(string pSamplePath, int pGrupo, int pLabel, string pFileName)
			{
				samplePath = pSamplePath;
				group = pGrupo;
				label = pLabel;
				fileName = pFileName;
			}
		} Sample;

Mat IntensityNormalization(Mat img);


double generateNoise(double tamanhoMaximo, int iteracao){
	if(iteracao > 0)
	{
		double g = rng.gaussian(tamanhoMaximo);
		if(abs(g) > tamanhoMaximo){
			return tamanhoMaximo;
		}
		else{
			return g;
		}
	}
	else
		return 0;
}

string saveSample(Sample sampleData, Mat image, string path, int id)
{
	stringstream ss;
	ss << "G" << sampleData.group << "/" << sampleData.fileName << "_";
	if( id < 10)
		ss << "00" << id;
	else if(id < 100)
		ss << "0" << id;
	else
		ss << id;
	ss << ".jpg";

	imwrite(path + ss.str(), image);

	ss << " " << sampleData.label << endl;

	return ss.str();
}

void GenerateIntensityNormalizedDatabase(){

	FILE *labelFile = fopen((originalDataFolder + "label.txt").c_str(),"r");
	FILE *labelOutput = fopen((syntheticDataFolder + "label.txt").c_str(), "w");
	int gaussianSize = 3;
	int maxSamples = 30;
	while(!feof(labelFile)){
		//Informações sobre o arquivo atual
		char lineCharacters[50], aux;
		int group = -1, fileClass = -1;
		string line;
		stringstream FullPath;
		Point2f OlhoEsquerdo, OlhoDireito;

		fscanf(labelFile, "%c%d%s %d\n",&aux, &group, lineCharacters, &fileClass);
		line.append(lineCharacters);

		FullPath << originalDataFolder << "G" << group << line;

		Mat original = imread(FullPath.str(), 0);
		cout << "G" << group << line << endl;
		for(int i = 0; i < maxSamples; i++){
			Mat img = NormalizeImageFile(FullPath.str(),
									Point2f(generateNoise(gaussianSize, i), generateNoise(gaussianSize, i)),
									Point2f(generateNoise(gaussianSize, i), generateNoise(gaussianSize, i)));

			string label = saveSample(Sample(FullPath.str(), group, fileClass, line.substr(1, line.size()-5)), img, syntheticDataFolder, i);
			fprintf(labelOutput, "%s", label.c_str());
		}
	}

	fclose(labelFile);
}

void GenerateNeutralSubtractedDatabase(int GaussianSize){
	//Parâmetros para geração dos exemplos sintéticos
	double gaussianSize = GaussianSize;
	int maxSamples = 50; // 16 + 1

	//Arquivos de entrada e saída, respectivamente
	FILE *labelFile = fopen((originalDataFolder + "label.txt").c_str(),"r");
	FILE *labelOutput = fopen((syntheticDataFolder + "label.txt").c_str(), "w");

	//Variavéis de controle na iteração dos exemplos
	string pessoaAnterior(""), baseImgPath;
	bool mudou = false;

	//Armazena, temporariamente, a imagens de uma pessoa
	Mat baseImg, imgDiferenca;
	queue<Sample> naoAnalisados;

	while(!feof(labelFile)){

		//Informações sobre o arquivo atual
		char lineCharacters[50], aux;
		int group = -1, fileClass = -1;
		string line;
		stringstream FullPath;
		//Point2f OlhoDireito, OlhoEsquerdo;
		fscanf(labelFile, "%c%d%s %d\n",&aux, &group, lineCharacters, &fileClass);

		line.append(lineCharacters);

		FullPath << originalDataFolder << "G" << group << line;

		//Verifica se foram percorridas todas as imagens de uma pessoa
		if (line.substr(1, 4) != pessoaAnterior || feof(labelFile)){
			pessoaAnterior = line.substr(1, 4);
			mudou = true;
			cout << pessoaAnterior << endl;
		}
		else{
			mudou = false;
		}

		//Se mudou a pessoa, gera todas as imagens da pessoa anterior
		if(mudou){
			while (naoAnalisados.size() > 0){
				Sample current = naoAnalisados.front();
				naoAnalisados.pop();
				for(int i = 0; i < maxSamples; i++){

						imgDiferenca = GenerateDifferenceImage(
										NormalizeImageFile(baseImgPath,
														  Point2f(generateNoise(gaussianSize, i), generateNoise(gaussianSize, i)),
														  Point2f(generateNoise(gaussianSize, i), generateNoise(gaussianSize, i))),
										NormalizeImageFile(current.samplePath,
														  Point2f(generateNoise(gaussianSize, i), generateNoise(gaussianSize, i)),
														  Point2f(generateNoise(gaussianSize, i), generateNoise(gaussianSize, i))));

						//Salva o arquivo em disco e retorna o label (caminho + classe)
						string label = saveSample(current, imgDiferenca, syntheticDataFolder, i);
						fprintf(labelOutput, "%s", label.c_str());
				}
			}
			baseImgPath = FullPath.str();
		}
		else
		{
			//Adiciona os arquivos da mesma pessoa para posterior análise
			naoAnalisados.push(Sample(FullPath.str(), group, fileClass, line.substr(1, line.size()-5)));
		}
	}
	fclose(labelFile);
	fclose(labelOutput);
}

int main(int argc, char** argv) {
	GenerateIntensityNormalizedDatabase();
	//GenerateNeutralSubtractedDatabase(2);
	return 0;
}

