/*
 * util.h
 *
 *  Created on: 22/06/2015
 *      Author: alopes
 */

#ifndef TOOLS_UTIL_H_
#define TOOLS_UTIL_H_

#include <cstring>
#include <map>
#include <string>
#include <string.h>
#include <vector>
#include <exception>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <fstream>
#include <time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <queue>
#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"
#include <boost/smart_ptr.hpp>
#include "glog/logging.h"

using namespace caffe;
using namespace boost;
using namespace std;
using namespace cv;

#define YXMin 1.3
#define YXMax 3.2
#define XXMin 1.2
#define XXMax 1.2

typedef
		struct
		DadosNormalizacao
		{
			float anguloRotacao;
			float distanciaOlhos;
			Point2f media;
		} DadosNormalizacao;

typedef
		struct
		DadosTreino
		{
			int grupoValidacao;
			int grupoTeste;
			vector<Mat> trainSamples;
			vector<int> trainLabels;
			vector<Mat> testSamples;
			vector<int> testLabels;
			vector<Mat> validationSamples;
			vector<int> validationLabels;
			map<string, int> IndividuosTeste;
		} DadosTreino;

void copyFile(string source, string destination);
void copyFile(char *source, char *dest);
vector<int> shuffleVectors(vector<Mat>& Samples, vector<int>& Labels);

Mat GerarImagemDiferenca(Mat ImagemBase, Mat Imagem){
	Mat newImg;
	convertScaleAbs(Imagem, Imagem, 0.5, 128);
	convertScaleAbs(ImagemBase, ImagemBase, 0.5, 0);

	subtract(Imagem, ImagemBase, newImg);
		
	return newImg;
}

void copyFile(string source, string destination)
{
	copyFile(strdup(source.c_str()), strdup(destination.c_str()));
}

int getValueAtIndex (int index, map<string, int> myMap){
    map<string, int>::const_iterator end = myMap.end();

    int counter = 0;
    for (map<string, int>::const_iterator it = myMap.begin(); it != end; ++it) {

        if (counter == index)
            return it->second;

        counter++;
    }
    return -1;
}

void copyFile(char *source, char *dest)
{
    int childExitStatus;
    pid_t pid;
    int status;
    if (!source || !dest) {
        /* handle as you wish */
    }

    pid = fork();

    if (pid == 0) { /* child */
        execl("/bin/cp", "/bin/cp", source, dest, (char *)0);
    }
    else if (pid < 0) {
        /* error - couldn't start process - you decide how to handle */
    }
    else {
        /* parent - wait for child - this has all error handling, you
         * could just call wait() as long as you are only expecting to
         * have one child process at a time.
         */
        pid_t ws = waitpid( pid, &childExitStatus, WNOHANG);
        if (ws == -1)
        { /* error - handle as you wish */
        }

        if( WIFEXITED(childExitStatus)) /* exit code in childExitStatus */
        {
            status = WEXITSTATUS(childExitStatus); /* zero is normal exit */
            /* handle non-zero as you wish */
        }
        else if (WIFSIGNALED(childExitStatus)) /* killed */
        {
        }
        else if (WIFSTOPPED(childExitStatus)) /* stopped */
        {
        }
    }
}


//Shuffle two vectors using the same order.
//Returns the shuffle order
vector<int> shuffleVectors(vector<Mat>& Samples, vector<int>& Labels)
{
	CHECK(Samples.size() == Labels.size()) <<	"The vectors must be of the same size";

	//Create a vector of indexes
	vector<int> indexes;
	indexes.reserve(Samples.size());
	for (int i = 0; i < Samples.size(); ++i)
		indexes.push_back(i);

	//Shuffle the indexes
	random_shuffle(indexes.begin(), indexes.end());

	//Create new vectors to store the shuffled data
	vector<Mat> shuffledSamples;
	vector<int> shuffledLabels;
	shuffledSamples.reserve(Samples.size());
	shuffledLabels.reserve(Labels.size());

	//Put the data shuffled in the vectors
	for(int i = 0; i < Samples.size(); i++)
	{
		shuffledSamples.push_back(Samples[indexes[i]]);
		shuffledLabels.push_back(Labels[indexes[i]]);
	}

	//Swap the content to get the shuffled data in the vectors passed by parameters
	Samples.swap(shuffledSamples);
	Labels.swap(shuffledLabels);

	return indexes;
}

DadosNormalizacao gerarDadosNormalizacao(Point2f pontoMedioOlhoDireito, Point2f pontoMedioOlhoEsquerdo)
{
	Point2f pontoMedioOlhos, vetorAngulo;

	//Calcula o ponto médio dos dois olhos
	pontoMedioOlhos.x = (pontoMedioOlhoDireito.x + pontoMedioOlhoEsquerdo.x) / 2;
	pontoMedioOlhos.y = (pontoMedioOlhoDireito.y + pontoMedioOlhoEsquerdo.y) / 2;

	//Calcula o vetor formado entre o ponto médio e o olho esquerdo
	vetorAngulo.x = pontoMedioOlhoDireito.x - pontoMedioOlhos.x;
	vetorAngulo.y = pontoMedioOlhoDireito.y - pontoMedioOlhos.y;

	DadosNormalizacao resultado;
	resultado.anguloRotacao = atan2(vetorAngulo.y, vetorAngulo.x);
	resultado.media = Point2f(pontoMedioOlhos.x, pontoMedioOlhos.y);
	resultado.distanciaOlhos = sqrt(pow(vetorAngulo.x, 2) + pow(vetorAngulo.y, 2));

	return resultado;
}

Mat normalizarImagem(Mat Img, DadosNormalizacao dadosNormalizacao)
{
	Mat rotationMatix = getRotationMatrix2D(Point(round(dadosNormalizacao.media.x), round(dadosNormalizacao.media.y)), dadosNormalizacao.anguloRotacao*180.0 / 3.141516, 1.0);

	int newSize = max(Img.cols, Img.rows);
	Mat newImg = Mat::zeros(Size(newSize, newSize), CV_8UC1);

	warpAffine(Img, newImg, rotationMatix, Size(newSize, newSize));
//	resize(Img, newImg, Size(newSize, newSize));
	Mat croppedImg = newImg(
							Range(dadosNormalizacao.media.y - dadosNormalizacao.distanciaOlhos * YXMin,
								  dadosNormalizacao.media.y + dadosNormalizacao.distanciaOlhos * YXMax),
						    Range(dadosNormalizacao.media.x - dadosNormalizacao.distanciaOlhos * XXMin,
						    	  dadosNormalizacao.media.x + dadosNormalizacao.distanciaOlhos * XXMax));

//	Mat croppedImg = newImg;
	resize(croppedImg, croppedImg, Size(32, 32));
	return croppedImg;
}

double stdDev(Mat Img, Mat Mean, Point2f Center, int kSize)
{
    kSize = kSize / 2;
    int mean = Mean.at<uchar>(Center);
    double somatorio = 0.0;
    int qtde = 0;
    for (int y = Center.y - kSize; y <= Center.y + kSize; y++)
    {
        for (int x = Center.x - kSize; x <= Center.x + kSize; x++)
        {
            if (x >= 0 && x < Img.cols && y >= 0 && y < Img.rows)
            {
                Point p(x, y);
                somatorio += pow(Img.at<uchar>(p) -mean, 2);
                qtde++;
            }
        }
    }

    if (somatorio == 0 || qtde == 0)
    {
        return 1;
    }
    else
    {
        return sqrt(somatorio / qtde);
    }

}

Mat equalizar(Mat Img)
{
    int kSize = 7;
    //int kSize = 29;
    Mat gaussianImg;
    int pixelOriginal = 0;
    int pixelGaussiana = 0;
    double desvioPadrao = 0;
    double novoPixel = 0;
    double newVal = 0;
    resize(Img, Img, Size(32, 32));

    Mat newImg = Mat::zeros(Size(Img.cols, Img.rows), CV_8UC1);

    GaussianBlur(Img, gaussianImg, Size(kSize, kSize), 0, 0);

    int qtde = 0;
    for (size_t y = 0; y < Img.rows; y++)
    {
        for (size_t x = 0; x < Img.cols; x++)
        {
            Point p(x, y);
            pixelOriginal = (int)Img.at<uchar>(p);
            pixelGaussiana = (int)gaussianImg.at<uchar>(p);
            desvioPadrao = stdDev(Img, gaussianImg, p, kSize * 2 + 1);

            novoPixel = (pixelOriginal - pixelGaussiana) / desvioPadrao;

            newVal = novoPixel * 127 / 2.0 + 127;

            if (newVal > 255)
            {
                qtde++;
                newVal = 255;
            }
            else if (newVal < 0)
            {
                newVal = 0;
                qtde++;
            }


            newImg.at<uchar>(p) = newVal;
        }
    }
    return newImg;
}

void cleanMatrix(int** mtz, int rows, int cols){

}

void trunkVectors(vector<Mat>& Imgs, vector<int>& Labels, int TrunkSize){
	while(Imgs.size() > TrunkSize){
		Imgs.erase(Imgs.begin());
		Labels.erase(Labels.begin());
	}
}

//Get the class used in the classifier
int getClassCK(int i)
{
	//As the class 2 isn't in the database, we get all classes higher than 1 (3, 4, 5,6,7) and deacreases, to get all classes in the interval 1-6
	if(i > 1){
		i--;
	}

	//Get the class 0-indexed
	return --i;
}

vector<Point2f> lerPontosLandmarkCohnKanade(string landmark)
{
        FILE* dadosImagem = fopen(landmark.c_str(), "r");
        float x, y;
        vector<Point2f> points;
        while (!feof(dadosImagem)){
                fscanf(dadosImagem, "%f %f\r\n", &x, &y);
                points.push_back(Point2f((int)x, (int)y));
                //points.push_back(Point2f((int)x*0.3, (int)y*0.3));
        //      cout << (int)x*0.3 << ", " << (int)y*0.3 << endl;
        }
        fclose(dadosImagem);

        return points;
}

DadosNormalizacao normalizarPontos(vector<Point2f>& lstPontos, Point2f RuidoOlhoDireito, Point2f RuidoOlhoEsquerdo)
{
        Point2f pontoMedioOlhoEsquerdo, pontoMedioOlhoDireito, pontoMedioOlhos, vetorAngulo;

        //Pontos do Olho Esquerdo
        for(int i = 36; i < 42; i++)
        {
                pontoMedioOlhoEsquerdo.x += lstPontos[i].x;
                pontoMedioOlhoEsquerdo.y += lstPontos[i].y;
        }
        pontoMedioOlhoEsquerdo.x /= 6;
        pontoMedioOlhoEsquerdo.y /= 6;

        //Adiciona o Ruido
        pontoMedioOlhoEsquerdo.x += RuidoOlhoEsquerdo.x;
        pontoMedioOlhoEsquerdo.y += RuidoOlhoEsquerdo.y;

        //Pontos do Olho Direito
        for(int i = 42; i < 48; i++)
        {
                pontoMedioOlhoDireito.x += lstPontos[i].x;
                pontoMedioOlhoDireito.y += lstPontos[i].y;
        }
        pontoMedioOlhoDireito.x /= 6;
        pontoMedioOlhoDireito.y /= 6;

        //Adiciona o Ruido
        pontoMedioOlhoDireito.x += RuidoOlhoDireito.x;
        pontoMedioOlhoDireito.y += RuidoOlhoDireito.y;
//Calcula o ponto médio dos dois olhos
        pontoMedioOlhos.x = (pontoMedioOlhoDireito.x + pontoMedioOlhoEsquerdo.x) / 2;
        pontoMedioOlhos.y = (pontoMedioOlhoDireito.y + pontoMedioOlhoEsquerdo.y) / 2;

        //Calcula o vetor formado entre o ponto médio e o olho esquerdo
        vetorAngulo.x = pontoMedioOlhoDireito.x - pontoMedioOlhos.x;
        vetorAngulo.y = pontoMedioOlhoDireito.y - pontoMedioOlhos.y;

        DadosNormalizacao resultado;
        resultado.anguloRotacao = atan2(vetorAngulo.y, vetorAngulo.x);
        resultado.media = Point2f(pontoMedioOlhos.x, pontoMedioOlhos.y);
        resultado.distanciaOlhos = sqrt(pow(vetorAngulo.x, 2) + pow(vetorAngulo.y, 2));

        return resultado;
}


Mat normalizarArquivo(string caminhoArquivo, Point2f RuidoOlhoDireito, Point2f RuidoOlhoEsquerdo)
{
        vector<Point2f> pontosImg = lerPontosLandmarkCohnKanade(caminhoArquivo.substr(0, caminhoArquivo.size() - 4).append("_landmarks.txt"));
//        vector<Point2f> pontosImg = lerPontosLandmarkCohnKanade(caminhoArquivo.append(".txt"));

        DadosNormalizacao resultadoNormalizacao = normalizarPontos(pontosImg, RuidoOlhoDireito, RuidoOlhoEsquerdo);
//cout << caminhoArquivo << endl;
        return normalizarImagem(imread(caminhoArquivo, 0), resultadoNormalizacao);

        //return normalizarImagem(imread(caminhoArquivo.substr(0, caminhoArquivo.size()-4), 0), resultadoNormalizacao);
}



#endif /* TOOLS_UTIL_H_ */
