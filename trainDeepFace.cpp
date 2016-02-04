#include "util.h"
#include "utilCaffe.h"

/*
 * 19/11/2015 - No treinamento Ã© escolhida a Ã©poca que deu o melhor resultado. Por isso
 * a necessidade de separamos os dados tem trÃªs grupos, treino, validaÃ§Ã£o e teste. O dado
 * de teste Ã© utilizado somente para avaliar a rede, apÃ³s todo o treinamento, que inclusive usa
 * a validaÃ§Ã£o para determinar a melhor rede e Ã©poca.
*/

int qtdeExpressoes = 7;
int qtdeGrupos = 10;
int qtdeIteracoes = 10;

string dataPath = "/dados/alopes/faces/sibgrapi-si/db-jaffe/";
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

int getClassJaffe(int i){
	if(qtdeExpressoes == 6)
		return --i;
	else
		return i;
}

void lerDadosTreino(DadosTreino* dadosTreino)
{
	char filename[100], charGrupo;
	int expressao, grupo;
	Mat fileImg;

	FILE *f = fopen((dataPath + "label.txt").c_str(), "r");

	while(!feof(f)){
		fscanf(f, "%c%d%s %d\n", &charGrupo, &grupo, filename, &expressao);
//		cout << filename << endl;
		if(expressao > 0 || qtdeExpressoes == 7){
			stringstream ss;
			ss << dataPath << "G" << grupo << filename;
//			cout << ss.str() << endl;
//			fileImg = imread(ss.str(), 0);
//			cout << "1" << endl;
			if(grupo == dadosTreino->grupoTeste){
				if(isOriginal(string(filename))){
					fileImg = imread(ss.str(),0);
					dadosTreino->testSamples.push_back(fileImg);
					dadosTreino->testLabels.push_back(getClassJaffe(expressao));
					string individuo(filename);
					individuo = individuo.substr(1, 4);
					if(dadosTreino->IndividuosTeste.find(individuo) == dadosTreino->IndividuosTeste.end())
					{
						dadosTreino->IndividuosTeste[individuo] = 0;
					}
					dadosTreino->IndividuosTeste[individuo]++;
				}
			}
			else if(grupo == dadosTreino->grupoValidacao){
				/* -- VERIFICAR SE Ã‰ MELHOR COLOCAR OU NÃƒO OS SINTÃ‰TICOS NO GRUPO DE VALIDAÃ‡ÃƒO --*/
				if(isOriginal(string(filename))){
					dadosTreino->validationSamples.push_back(fileImg);
					dadosTreino->validationLabels.push_back(getClassJaffe(expressao));
				}
			}
			else{
				if(fileImg.rows > 0){
	//				dadosTreino->trainSamples.push_back(fileImg);
	//				dadosTreino->trainLabels.push_back(getClassJaffe(expressao));
				}
			}
		}
	}
	fclose(f);
	//Faz com que o conjunto seja divisivel pelo tamanho do batch
	shuffleVectors(dadosTreino->validationSamples, dadosTreino->validationLabels);
	while(dadosTreino->validationSamples.size() % 20 != 0){
		dadosTreino->validationSamples.erase(dadosTreino->validationSamples.begin());
		dadosTreino->validationLabels.erase(dadosTreino->validationLabels.begin());
	}

	shuffleVectors(dadosTreino->trainSamples, dadosTreino->trainLabels);
	while(dadosTreino->trainSamples.size() % 50 != 0){
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
	Teste do treinamento do JAFFE usando somente treino e teste. Um dos motivos e o BD ser pequeno. 
	O outro eh que o resultado na outra metodologia nao foi muito satisfatorio. Neste caso foi tirado
	a selecao da melhor epoca durante o treinamento.
*/
void rodarExperimentoJaffe(){
	string solver = dataPath + "solver.prototxt";
        string resultsPath = dataPath + "Results-Average2/";

        for(int grupoTeste = 1; grupoTeste <= qtdeGrupos; grupoTeste++){
        	DadosTreino dadosTreino;
                dadosTreino.grupoTeste = grupoTeste;
                dadosTreino.grupoValidacao = 99;
                lerDadosTreino(&dadosTreino);
 		while(dadosTreino.testSamples.size() % 10 != 0){
                        dadosTreino.testSamples.erase(dadosTreino.testSamples.begin());
                	dadosTreino.testLabels.erase(dadosTreino.testLabels.begin());
                }

		DadosTreino dadosTeste;
		dadosTeste.grupoTeste = grupoTeste;
		dadosTeste.grupoValidacao = 99;
		lerDadosTreino(&dadosTeste);

                for(int iter = 1; iter <= qtdeIteracoes; iter++){
                	shuffleVectors(dadosTreino.trainSamples, dadosTreino.trainLabels);
                        trainNet(solver, dadosTreino.trainSamples, dadosTreino.trainLabels, dadosTreino.testSamples, dadosTreino.testLabels);
                        executarTeste(resultsPath, grupoTeste, grupoTeste, iter, dadosTeste.testSamples, dadosTeste.testLabels, -1, "facesnet_iter_10000.caffemodel");
                }
        }
}

/*
	Experimento usando 7 grupos para treino, 1 grupo para validacao e teste. Este ultimo eh dividido
	em dois grupos, um com (n - 1) individuos para validacao e (1) individuo para teste.
*/
void rodarExperimentoDoisGrupos(){
	string solver = dataPath + "solver.prototxt";
	string resultsPath = dataPath + "sibgrapi7/";

	for(int grupoTeste = 1; grupoTeste <= qtdeGrupos; grupoTeste++){
		DadosTreino dadosTreino;
		dadosTreino.grupoTeste = grupoTeste;
		//Garante que so vai separar em treino e teste, nao vai ter validacao
		dadosTreino.grupoValidacao = 99;
		lerDadosTreino(&dadosTreino);
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
			while(imagensValidacao.size() % 20 != 0){
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

/*void rodarExperimentoCrossBank(){
	string solver = dataPath + "solver.prototxt";

	dataPath = "/dados/alopes/faces/JAFFE/Sinteticos-Grupos/";
	DadosTreino dadosOutroBd;
	dadosOutroBd.grupoTeste = 99;
	for(int i = 1; i <= 10; i++){
		dadosOutroBd.grupoTeste = i;
		lerDadosTreino(&dadosOutroBd);
		while(dadosOutroBd.trainSamples.size() > 0){
			dadosOutroBd.trainSamples.erase(dadosOutroBd.trainSamples.begin());
			dadosOutroBd.trainLabels.erase(dadosOutroBd.trainLabels.begin());
		}
	}

	dataPath = "/dados/alopes/faces/CK/Sinteticos-Grupos/";
	string resultsPath = dataPath + "Results-Cross-Jaffe/";

        for(int grupoTeste = 1; grupoTeste <= qtdeGrupos; grupoTeste++){
                DadosTreino dadosTreino;
                dadosTreino.grupoTeste = grupoTeste;
                dadosTreino.grupoValidacao = 99;
                lerDadosTreino(&dadosTreino);
                while(dadosTreino.testSamples.size() % 20 != 0){
                        dadosTreino.testSamples.erase(dadosTreino.testSamples.begin());
                        dadosTreino.testLabels.erase(dadosTreino.testLabels.begin());
                }

                DadosTreino dadosTeste;
                dadosTeste.grupoTeste = grupoTeste;
                dadosTeste.grupoValidacao = 99;
                lerDadosTreino(&dadosTeste);
		double bstValidation = 0;
                for(int iter = 1; iter <= qtdeIteracoes; iter++){
                        shuffleVectors(dadosTreino.trainSamples, dadosTreino.trainLabels);
                        trainNet(solver, dadosTreino.trainSamples, dadosTreino.trainLabels, dadosTreino.testSamples, dadosTreino.testLabels);
			double validation =  testNet(dataPath + "test.prototxt", defaultNetFile, dadosTeste.testSamples, dadosTeste.testLabels, NULL, dataPath + "solver.prototxt");
			if(validation > bstValidation){
	                        executarTeste(resultsPath, grupoTeste, grupoTeste, 0, dadosOutroBd.testSamples, dadosOutroBd.testLabels, -1);
			}
                }
        }
}*/

void rodarExperimentoCrossBank(){
	FILE *redes = fopen("/home/likewise-open/LCAD/alopes/Resultados/subtracao7/redes.txt","r");

	dataPath = "/dados/alopes/faces/JAFFE/Sinteticos-Grupos/";
        DadosTreino dadosOutroBd;
        dadosOutroBd.grupoTeste = 99;
        for(int i = 1; i <= 10; i++){
                dadosOutroBd.grupoTeste = i;
                lerDadosTreino(&dadosOutroBd);
                while(dadosOutroBd.trainSamples.size() > 0){
                        dadosOutroBd.trainSamples.erase(dadosOutroBd.trainSamples.begin());
                        dadosOutroBd.trainLabels.erase(dadosOutroBd.trainLabels.begin());
                }
//		cout << "G" << i <<  endl;
        }


	int i = 1;
	while(!feof(redes)){
		char rede[100];
		fscanf(redes, "%s", rede);
		stringstream ss;
		ss << "/home/likewise-open/LCAD/alopes/Resultados/subtracao7/";
		ss << rede;

		executarTeste("/home/likewise-open/LCAD/alopes/Resultados/subtracao-cross-jaffe7/", 0, 0, i, dadosOutroBd.testSamples, dadosOutroBd.testLabels, -1, ss.str());
		i++;
	}
}



/*
	Experimento Usando 6 grupos para treino, 1 para validacao e 1 para teste.
	Esse teste pode ser realizado de duas formas:
	1. 	Usando o grupo de validacao para determinar a melhor epoca durante o treino. 
		E testar todas as 10 rodadas no grupo de teste.
	2. 	Usando o grupo de validacao para determinar a melhor epoca e a melhor das 10 rodadas.
		E testar somente a melhor rodada no grupo de teste.
*/
void rodarExperimentoTresGrupos(){
	string solver = dataPath + "solver.prototxt";
	string resultsPath = dataPath + "sibgrapi-sintetico-sem-equalizacao/";

	for(int grupoTeste = 1; grupoTeste <= 1; grupoTeste++){
//		for(int grupoValidacao = 1; grupoValidacao <= qtdeGrupos; grupoValidacao++){
//			if(grupoValidacao != grupoTeste){
				DadosTreino dadosTreino;
				dadosTreino.grupoTeste = grupoTeste;
				dadosTreino.grupoValidacao = 99;
				lerDadosTreino(&dadosTreino);
				 while(dadosTreino.testSamples.size() % 10 != 0){
			                dadosTreino.testSamples.erase(dadosTreino.testSamples.begin());
			                dadosTreino.testLabels.erase(dadosTreino.testLabels.begin());
			        }
//	cout << dadosTreino.testSamples.size() << " - " << dadosTreino.trainSamples.size() << endl;
//getchar();

				for(int iter = 1; iter <= 1; iter++){
//					shuffleVectors(dadosTreino.trainSamples, dadosTreino.trainLabels);
					trainNet(solver, dadosTreino.trainSamples, dadosTreino.trainLabels, dadosTreino.testSamples, dadosTreino.testLabels);
					executarTeste(resultsPath, grupoTeste, grupoTeste, iter, dadosTreino.testSamples, dadosTreino.testLabels);
				}
			}
//		}
//	}
}

void encontrarMelhorRedeGrupo(string resultsPath){
        string solver = dataPath + "solver.prototxt";

    for(int grupoTeste = 1; grupoTeste <= qtdeGrupos; grupoTeste++){
                for(int grupoValidacao = 1; grupoValidacao <= qtdeGrupos; grupoValidacao++){
                        if(grupoValidacao != grupoTeste){
                                double bstValidationAccuracy = 0;
                                DadosTreino dadosTreino;
                  
        dadosTreino.grupoTeste = grupoTeste;
                                dadosTreino.grupoValidacao = grupoValidacao;
                                lerDadosTreino(&dadosTreino);
                                for(int iter = 1; iter <= qtdeIteracoes; iter++){
                                        stringstream ss;
                                        ss << resultsPath << "summary_GT" << grupoTeste << "_GV" << grupoValidacao << "_IT" << iter << ".net";
                                        double validationAccuracy = testNet(dataPath + "test.prototxt", ss.str(), dadosTreino.validationSamples, dadosTreino.validationLabels, NULL, solver);
                                        if(validationAccuracy > bstValidationAccuracy){
                                                bstValidationAccuracy = validationAccuracy;
                                                executarTeste(resultsPath, grupoTeste, grupoValidacao, 0, dadosTreino.testSamples, dadosTreino.testLabels, -1 ,ss.str());
                                        }
                                }
                        }
                }
        }
}

void testarTodasRedes(string resultsPath){
    string solver = dataPath + "solver.prototxt";

    for(int grupoTeste = 1; grupoTeste <= qtdeGrupos; grupoTeste++){
		for(int grupoValidacao = 1; grupoValidacao <= qtdeGrupos; grupoValidacao++){
			if(grupoValidacao != grupoTeste){
				double bstValidationAccuracy = 0;
				DadosTreino dadosTreino;
				dadosTreino.grupoTeste = grupoTeste;
				dadosTreino.grupoValidacao = grupoValidacao;
				lerDadosTreino(&dadosTreino);
				for(int iter = 1; iter <= qtdeIteracoes; iter++){
					stringstream ss;
					ss << resultsPath << "summary_GT" << grupoTeste << "_GV" << grupoValidacao << "_IT" << iter << ".net";
					//double validationAccuracy = testNet(dataPath + "test.prototxt", ss.str(), dadosTreino.validationSamples, dadosTreino.validationLabels, NULL, solver);
					//if(validationAccuracy > bstValidationAccuracy){
					//	bstValidationAccuracy = validationAccuracy;
						executarTeste(resultsPath, grupoTeste, grupoValidacao, iter, dadosTreino.testSamples, dadosTreino.testLabels, -1 ,ss.str());
				//	}
				}
			}
		}
	}
}

void encontrarMelhorRedeIndividuo(string resultsPath){
	string solver = dataPath + "solver.prototxt";

	for(int grupoTeste = 1; grupoTeste <= 8; grupoTeste++){
		DadosTreino dadosTreino;
		dadosTreino.grupoTeste = grupoTeste;
		//Garante que so vai separar em treino e teste, nai vai ter validacao
		dadosTreino.grupoValidacao = 99;
		lerDadosTreino(&dadosTreino);
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
					executarTeste(resultsPath, grupoTeste, grupoTeste, 0, imagensTeste, labelsTeste, individuo, "/home/likewise-open/LCAD/alopes/Resultados/sibgrapi7/summary_GT8_GV8_IT9_IND14.net");
				}				
			}
		}
	}
}

int main(){
//	encontrarMelhorRedeIndividuo("/home/likewise-open/LCAD/alopes/Resultados/sibgrapi7/");
	//encontrarMelhorRedeGrupo(dataPath + "subtracao7/");
//	rodarExperimentoDoisGrupos();
//	rodarExperimentoTresGrupos();
	rodarExperimentoCrossBank();
	//rodarExperimentoJaffe();
}
