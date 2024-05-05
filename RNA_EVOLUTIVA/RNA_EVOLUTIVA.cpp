/*
  Rede Neural Artificial Evolutiva (RNA-E)

  Os pesos s�o atualizados a partir de um algoritmo
  gen�tico que busca minimizar os erros na fase de
  treinamento.

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_Entradas 2
#define MAX_Pesos 6

//===| Estrutura de Dados |==========================
typedef char string[60];

typedef struct tipoLicao
{
	int p;				   // proposi��o P
	int q;				   // Proposi��o Q
	int resultadoEsperado; // Proposi��o Composta P "E" Q (A Classe)
	tipoLicao *prox;
} TLicao;

typedef struct tipoIndividuo
{
	float genes[MAX_Pesos];
	int erros;
	int numero; // numero identificador
	tipoIndividuo *prox;
} TIndividuo;

typedef struct tipoSinapse
{
	int camada;
	int neuronio_origem;
	int neuronio_destino;
	float peso;
	tipoSinapse *prox;
} TSinapse;

typedef struct tipoNeuronio
{
	int neuronio;
	float soma;
	float peso;
	tipoNeuronio *prox;
} TNeuronio;

typedef struct tipoLista
{
	FILE *fp; // Arquivo de Sa�da (Relat�rio).
	string objetivo;
	TLicao *licoes; // Conjunto de li��es a serem aprendidas
	float entradas[MAX_Entradas];
	TNeuronio *neuronios;
	TSinapse *pesos;
	TIndividuo *populacao;
	TIndividuo *individuoAtual;
	int totalIndividuos;
	int Qtd_Populacao;
	int Qtd_Mutacoes_por_vez;
	int Total_geracoes;
	int geracao_atual;
	int Qtd_Geracoes_para_Mutacoes;
	float sinapseThreshold;
	float learningRate;
} TLista;

TLista lista;

//====| Assinatura de Fun��es |=======================
void inicializa(TLista *L);
void geraIndividuos(TLista *L);
void geraLicoes(TLista *L);
void insereLicao(TLista *L, int p, int q, int resultado);
void insereNeuronio(TLista *L, int neuronio);
void estabelecendoSinapse(TLista *L, int neuronioDe, int neuronioAte, int camada);
void treinamento(TLista *L);
void cruzamento(TLista *L);
void avaliacaoIndividuos(TLista *L);
void ordenamentoIndividuos(TLista *L);
void promoveMutacoes(TLista *L);
void poda(TLista *L);
//===| Programa Principal |===========================
int main()
{
	inicializa(&lista);
	treinamento(&lista);
}
//===| Fun��es |======================================
void inicializa(TLista *L)
{
	int i;

	L->licoes = NULL;
	L->populacao = NULL;
	L->individuoAtual = NULL;
	L->totalIndividuos = 0;

	for (i = 0; i < MAX_Entradas; i++)
	{
		L->entradas[i] = 0;
	} // for

	L->neuronios = NULL;
	L->pesos = NULL;

	printf("\t\t=====| REDE NEURAL ARTIFICIAL EVOLUTIVA |=====");
	printf("\n\n\t\t=====| Configuracao da RNA |=====\n\n");
	printf("\tInforme o TAMANHO da POPULACAO (em termos de individuos):\n");
	printf("\t\tSugestao: 300 individuos.\n\t\tValor: ");
	scanf("%d", &L->Qtd_Populacao);

	geraIndividuos(L);

	printf("\n\n\tInforme a QUANTIDADE de GERACOES maxima:");
	printf("\n\tSugestao: 100 geracoes no total.\n\tValor: ");
	scanf("%d", &L->Total_geracoes);

	L->geracao_atual = 0;

	printf("\n\n\tInforme o INTERVALO de GERACOES para a ocorrencia de MUTACOES:");
	printf("\n\tSugestao: 5 (a cada 5 geracoes devem ocorrer mutacoes).\n\tValor: ");
	scanf("%d", &L->Qtd_Geracoes_para_Mutacoes);

	printf("\n\n\tInforme a QUANTIDADE de MUTACOES que devem ocorrer POR VEZ:");
	printf("\n\tSugestao: 3 mutacoes por intervalo.\n\tValor: ");
	scanf("%d", &L->Qtd_Mutacoes_por_vez);

	printf("\n\nSINAPSE THRESHOLD (Limiar das Conexoes entre Neuronios):\n");
	printf("Define a intensidade do sinal que sensibiliza cada neuronio.\n\n");
	printf("\tInforme o SINAPSE THRESHOLD:\n\tSugestao: 0.60\n\tValor: ");
	scanf("%f", &L->sinapseThreshold);

	printf("\n\nLEARNING RATE (Taxa de Aprendizado): variacao dos pesos em cada ajuste (Aprendizado).\n");
	printf("\n\tLEARNING RATE:\n\tSugestao: 0.20\n\tValor: ");
	scanf("%f", &L->learningRate);

	strcpy(L->objetivo, "Aprendizado da Funcao Logica P E Q");

	printf("\n\n\tDefinindo as LICOES a serem aprendidas pela Rede Neural Artificial.\n\n");
	geraLicoes(L);

	printf("\n\n\tDefinindo os NEURONIOS que compoem a REDE NEURAL ARTIFICIAL.");
	insereNeuronio(L, 1);
	insereNeuronio(L, 2);
	insereNeuronio(L, 3);
	insereNeuronio(L, 4);
	insereNeuronio(L, 5);

	printf("\n\n\tEstabelecendo as CONEXOES (Sinapses) entre os NEURONIOS.");
	estabelecendoSinapse(L, 1, 3, 0);
	estabelecendoSinapse(L, 2, 3, 0);
	estabelecendoSinapse(L, 2, 4, 0);
	estabelecendoSinapse(L, 3, 5, 1);
	estabelecendoSinapse(L, 4, 5, 1);

	L->fp = fopen("RNA_EVOLUTIVA_RELATORIO.TXT", "w");

	fprintf(L->fp, "\n\t\t=====| REDE NEURAL ARTIFICIAL EVOLUTIVA |=====\n\n");
	fprintf(L->fp, "\tOBJETIVO: %s.\n\n\tLicoes:\n", L->objetivo);
	fprintf(L->fp, "\t LICAO    P    Q  (Resultado Esperado)\n");
	fprintf(L->fp, "\t+------+----+----+---------------------+\n");

	TLicao *licao = L->licoes;
	int cont = 0;
	while (licao != NULL)
	{
		fprintf(L->fp, "\t(%d) - %d   %d   %d\n", ++cont, licao->p, licao->q, licao->resultadoEsperado);
		licao = licao->prox;
	} // while

	fprintf(L->fp, "\n\n");
	fprintf(L->fp, "\tLearning Rate: %.2f\n", L->learningRate);
	fprintf(L->fp, "\tSinapse Threshold: %.2f\n", L->sinapseThreshold);
	fprintf(L->fp, "\tPopulacao MAXIMA: %d.\n", L->Qtd_Populacao);
	fprintf(L->fp, "\t%d MUTACOES a cada sequencia de %d GERACOES.\n", L->Qtd_Mutacoes_por_vez, L->Qtd_Geracoes_para_Mutacoes);
	fprintf(L->fp, "\tTOTAL de GERACOES: %d.\n\n\n", L->Total_geracoes);

	printf("\n\n\tConfiguracao FINALIZADA!!!\n\n");
}
//====================================================
void geraIndividuos(TLista *L)
{
	TIndividuo *novo;
	int i, x;

	srand((unsigned)time(NULL));

	for (i = 0; i < L->Qtd_Populacao; i++)
	{
		novo = (TIndividuo *)malloc(sizeof(TIndividuo));

		novo->prox = NULL;
		novo->numero = i + 1;
		novo->erros = -1;

		for (x = 0; x < MAX_Pesos; x++)
		{
			novo->genes[x] = rand() % 101;
			novo->genes[x] = novo->genes[x] / 100;
		} // for

		if (L->populacao == NULL)
		{
			L->populacao = novo;
		}
		else
		{
			TIndividuo *atual = L->populacao;

			while (atual->prox != NULL)
			{
				atual = atual->prox;
			} // while

			atual->prox = novo;
		} // if

		L->totalIndividuos++;
	} // for
}
//=====================================================
void geraLicoes(TLista *L)
{
	TLicao *novo;
	int p, q;

	insereLicao(L, 0, 0, 0);
	insereLicao(L, 0, 1, 0);
	insereLicao(L, 1, 0, 0);
	insereLicao(L, 1, 1, 1);
}
//=====================================================
void insereLicao(TLista *L, int p, int q, int resultado)
{
	TLicao *novo = (TLicao *)malloc(sizeof(TLicao));

	novo->prox = NULL;
	novo->p = p;
	novo->q = q;
	novo->resultadoEsperado = resultado;

	if (L->licoes == NULL)
	{
		L->licoes = novo;
	}
	else
	{
		TLicao *atual = L->licoes;

		while (atual->prox != NULL)
		{
			atual = atual->prox;
		} // while
		atual->prox = novo;
	} // if
}
//======================================================
void insereNeuronio(TLista *L, int neuronio)
{
	TNeuronio *novo = (TNeuronio *)malloc(sizeof(TNeuronio));
	novo->prox = NULL;
	novo->neuronio = neuronio;
	novo->peso = 0;
	novo->soma = 0;

	if (L->neuronios == NULL)
	{
		L->neuronios = novo;
	}
	else
	{
		TNeuronio *atual = L->neuronios;

		while (atual->prox != NULL)
		{
			atual = atual->prox;
		} // while
		atual->prox = novo;
	} // if
}
//======================================================
void estabelecendoSinapse(TLista *L, int neuronioDe, int neuronioAte, int camada)
{
	TSinapse *novo = (TSinapse *)malloc(sizeof(TSinapse));
	TSinapse *atual;

	novo->prox = NULL;
	novo->neuronio_origem = neuronioDe;
	novo->neuronio_destino = neuronioAte;
	novo->camada = camada;
	novo->peso = 0;

	if (L->pesos == NULL)
	{
		L->pesos = novo;
	}
	else
	{
		atual = L->pesos;

		while (atual->prox != NULL)
		{
			atual = atual->prox;
		} // while
		atual->prox = novo;
	} // if
}
//=============================================================
void treinamento(TLista *L)
{
	printf("\n\n\t\t=====| INICIADO TREINAMENTO |=====\n\n");
	fprintf(L->fp, "\n\n\tINICIO DO TREINAMENTO: ");
	// ponteiro para a struct que armazena data e hora:
	struct tm *data_hora_atual;
	// vari�vel do tipo time_t para armazenar o tempo em segundos.
	time_t segundos;
	// Obetendo o tempo em segundos.
	time(&segundos);
	// Para converter de segundos para o tempo local
	// utilizamos a fun��o localtime().
	data_hora_atual = localtime(&segundos);

	fprintf(L->fp, "Dia: %d", data_hora_atual->tm_mday);
	fprintf(L->fp, "   Mes: %d", data_hora_atual->tm_mon);
	fprintf(L->fp, "   Ano: %d\n\n", data_hora_atual->tm_year);

	fprintf(L->fp, "Dia da Semana: %d.\n", data_hora_atual->tm_wday);

	fprintf(L->fp, "%d", data_hora_atual->tm_hour);
	fprintf(L->fp, ":%d", data_hora_atual->tm_min);
	fprintf(L->fp, ":%d.\n\n", data_hora_atual->tm_sec);

	printf("antes do for\n");
	printf("Geracoes: %d\n", L->Total_geracoes);
	int i;
	for (i = 0; i < L->Total_geracoes; i++)
	{
		printf("Cruzamento\n");
		cruzamento(L);

		if ((i % L->Qtd_Geracoes_para_Mutacoes) == 0)
		{
			printf("mutacoes\n");
			promoveMutacoes(L);
		} // if

		printf("avaliacao\n");
		avaliacaoIndividuos(L);

		printf("ordenamento\n");
		ordenamentoIndividuos(L);

		printf("poda %d\n", i);
		poda(L);
		L->individuoAtual = L->populacao;
		int i;
		for (i = 0; i < L->Qtd_Populacao; i++)
		{
			printf("INDIVIDUO: %d\n", i + 1);
			printf("Genes: [ ");
			int h;
			for (h = 0;h < 6;h++)
			{
				printf("%.2f ", L->individuoAtual->genes[h]);
			}
			printf(" ]");
			printf("\nErros: %d\n\n", L->individuoAtual->erros);
			L->individuoAtual = L->individuoAtual->prox;
		}

	} // for
}
void insereIndividuo(TLista *L, TIndividuo *individuo)
{
	TIndividuo *atual = L->populacao;
	while (atual->prox != NULL)
	{
		atual = atual->prox;
	}
	atual->prox = individuo;
	atual->prox->erros = -1;
	atual->prox->prox = NULL;
}
int randomOutOfFive(int x)
{
	int y = rand() % x;
	return y;
};
int maisouMenos()
{
	return rand() % 2;
}
//=============================================================
void cruzamento(TLista *L)
{
	/* Fun��o respons�vel pelo cruzamento de individuos.
	   Cada casal (selecionado por proximidade) gera dois
	   descendentes. E cada descendente herda segmentos
	   do c�digo gen�tico de seus pais.
	*/
	TIndividuo *pai = (TIndividuo *)malloc(sizeof(TIndividuo));
	TIndividuo *mae = (TIndividuo *)malloc(sizeof(TIndividuo));
	L->individuoAtual = L->populacao;
	int i;
	printf("ESTOU AQUI CRUZAMENTO\n");
	for (i = 0; i < L->Qtd_Populacao / 2; i++)
	{
		pai = L->individuoAtual;
		L->individuoAtual = L->individuoAtual->prox;
		mae = L->individuoAtual;

		TIndividuo *filho1 = (TIndividuo *)malloc(sizeof(TIndividuo));
		TIndividuo *filho2 = (TIndividuo *)malloc(sizeof(TIndividuo));
		L->totalIndividuos = L->totalIndividuos + 1;
		filho1->numero = L->totalIndividuos;

		filho1->genes[0] = pai->genes[0];
		filho1->genes[1] = pai->genes[1];
		filho1->genes[2] = pai->genes[2];
		filho1->genes[3] = mae->genes[0];
		filho1->genes[4] = mae->genes[1];
		filho1->genes[5] = mae->genes[2];
		insereIndividuo(L, filho1);

		L->totalIndividuos = L->totalIndividuos + 1;

		filho2->numero = L->totalIndividuos;
		filho2->genes[0] = pai->genes[3];
		filho2->genes[1] = pai->genes[4];
		filho2->genes[2] = pai->genes[5];
		filho2->genes[3] = mae->genes[3];
		filho2->genes[4] = mae->genes[4];
		filho2->genes[5] = mae->genes[5];
		insereIndividuo(L, filho2);

		L->individuoAtual = L->individuoAtual->prox;
	}
}

//=============================================================
void avaliacaoIndividuos(TLista *L)
{
	/*
	Avalia o grau de adapta��o de cada indiv�duo ao ambiente
	em termos de quantidade de erros cometidos nas li��es da
	RNA. O objetivo � MINIMIZAR esses ERROS at� ZERO.
	*/
	L->individuoAtual = L->populacao;
	TLicao *atualLicao;
	// camada: 0
	float n1, n2;
	// camada: 1
	float n3, soma3, peso13, peso23;
	float n4, soma4, peso14, peso24;
	// camada: 2
	float n5, soma5, peso35, peso45;
	while (L->individuoAtual != NULL)
	{
		if (L->individuoAtual->erros == -1)
		{
			L->individuoAtual->erros = 0;

			atualLicao = L->licoes;
			while (atualLicao != NULL)
			{
				// montagem da rede neural
				// camada 0
				n1 = atualLicao->p;
				n2 = atualLicao->q;

				peso13 = L->individuoAtual->genes[0];
				peso23 = L->individuoAtual->genes[2];

				peso14 = L->individuoAtual->genes[1];
				peso24 = L->individuoAtual->genes[3];
				// calculos
				soma3 = (n1 * peso13) + (n2 * peso23);
				soma4 = (n1 * peso14) + (n2 * peso24);
				// teste primeira camada
				soma3 >= L->sinapseThreshold ? n3 = 1 : n3 = 0;
				soma4 >= L->sinapseThreshold ? n4 = 1 : n4 = 0;
				// segunda camada
				peso35 = L->individuoAtual->genes[4];
				peso45 = L->individuoAtual->genes[5];
				// calculos
				soma5 = (n3 * peso35) + (n4 * peso45);

				soma5 >= L->sinapseThreshold ? n5 = 1 : n5 = 0;

				if (n5 != atualLicao->resultadoEsperado)
				{
					L->individuoAtual->erros++;
				}
				atualLicao = atualLicao->prox;
			}
		}
		L->individuoAtual = L->individuoAtual->prox;
	}
}
//==============================================================
void ordenamentoIndividuos(TLista *L)
{
	/* Reordena os indiv�duos por ordem ascendente de erros:
	   os indiv�duos que cometeram menos erros dever�o permanecer
	   no in�cio da Lista e os que cometeram mais erros dever�o
	   ficar no final da mesma Lista. */

	TIndividuo *atual = L->populacao;
	TIndividuo *prox = atual->prox;

	TSinapse *atualS = L->pesos;
	TSinapse *proxS = atualS->prox;

	TNeuronio *atualN = L->neuronios;
	TNeuronio *proxN = atualN->prox;

	int i, j;
	for (i = 0; i < L->Qtd_Populacao; i++)
	{
		for (j = 0; j < L->Qtd_Populacao - 1; j++)
		{
			if (atual > prox)
			{
				atual->prox = prox->prox;
				prox->prox = atual;

				atualS->prox = proxS->prox;
				proxS->prox = atualS;

				atualN->prox = proxN->prox;
				proxN->prox = atualN;

				if (j == 0)
				{
					L->populacao = prox;
					L->pesos = proxS;
					L->neuronios = proxN;
				}
			}
		}
	}
}
//==============================================================
void promoveMutacoes(TLista *L)
{
	/* Altera o c�digo gen�tico de um n�mero espec�fico
	   de indiv�duos (= L->Qtd_Mutacoes_por_vez). */
	int i;
	for (i = 0; i < L->Qtd_Mutacoes_por_vez; i++)
	{
		L->individuoAtual = L->populacao;
		int h;
		for (h = 0; h < (rand() % (L->Qtd_Populacao + 1)); h++)
		{
			L->individuoAtual = L->individuoAtual->prox;
		}
		int moum = maisouMenos();
		if (moum == 1)
		{
			L->individuoAtual->genes[randomOutOfFive(5)]++;
		}
		else
		{
			L->individuoAtual->genes[randomOutOfFive(5)]--;
		}
	}
}
//==============================================================
void poda(TLista *L)
{
	/* Elimina os indiv�duos menos aptos (que est�o no
	   fim da Lista) at� que a popula��o volte ao seu
	   Limite estabelecido na configura��o inicial
	   (L->Qtd_Populacao). */

	TIndividuo *atual = L->populacao;
	int i;
	for (int i = 0; i < L->Qtd_Populacao - 1; i++)
	{
		atual = atual->prox;
	}

	free(atual->prox);
	atual->prox = NULL;
	L->totalIndividuos = L->Qtd_Populacao;
}
//==============================================================
