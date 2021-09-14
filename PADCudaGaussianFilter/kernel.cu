#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_NAME 256 /* tamanho maximo para nome de arquivo */
#define PI 3.14159265359

double **matrizpesos;

struct param{
	int totalthreads;
	int linhas;
	int colunas;
	int tamanho; //se a divisao nao for inteira aqui vai o mais 1
	int sobra; //resto da divisao que vamos distribuir
	int raio; //distancia para considerar os pixels
};

void InicializaMatrizPesos(int raio);
double **AlocaMatrizDouble(int lin, int col);
double **LiberaMatrizDouble(int lin, int col, double **mat);
int **AlocaMatriz(int lin, int col);
int **LiberaMatriz(int lin, int col, int **mat);

__global__ void filtra(int **cudaOldimage, int **cudaNewimage, double **cudaMatrizPesos, param parameters){
	int linhai, linhaf, deslocamento, meutam, index;

	index = threadIdx.x + blockIdx.x * blockDim.x; //index global da thread
	meutam = parameters.tamanho;
	deslocamento = 0;

	if (parameters.sobra > 0){
		if (index < parameters.sobra){
			meutam++;
		}
		else {
			deslocamento = parameters.sobra;
		}
	}
	linhai = (meutam)* index + deslocamento;
	if (index == parameters.totalthreads - 1){
		linhaf = parameters.linhas - 1;
	}
	else {
		linhaf = (meutam)+linhai - 1;
	}
	//printf("Eu sou a thread %d fico com: linha %d a %d\n", index, linhai, linhaf);

	//tratamento_normal(linhai, linhaf);
	int dls, dli, dce, dcd;
	int r, g, b, nr, ng, nb;
	int l;
	int c;
	int lin_mat_pes, col_mat_pes;
	double acumular, acumulag, acumulab;
	for (l = linhai; l <= linhaf; l++){
		if ((l - parameters.raio) < 0){
			dls = parameters.raio - l;
		}
		else dls = 0;
		if ((l + parameters.raio) >= parameters.linhas){
			dli = l + parameters.raio - (parameters.linhas - 1);
		}
		else dli = 0;
		//printf("Linha: %d ... dls(%d) e dli(%d)\n", l, dls, dli);
		for (c = 0; c < parameters.colunas; c++){
			acumular = 0;
			acumulag = 0;
			acumulab = 0;
			if ((c - parameters.raio) < 0){
				dce = parameters.raio - c;
			}
			else dce = 0;
			if ((c + parameters.raio) >= parameters.colunas){
				dcd = c + parameters.raio - (parameters.colunas - 1);
			}
			else dcd = 0;
			for (lin_mat_pes = dls; lin_mat_pes < (parameters.raio * 2 + 1 - dli); lin_mat_pes++){
				for (col_mat_pes = dce; col_mat_pes < (parameters.raio * 2 + 1 - dcd); col_mat_pes++){
					r = cudaOldimage[l - parameters.raio + lin_mat_pes][c - parameters.raio + col_mat_pes] / 1000000;
					g = (cudaOldimage[l - parameters.raio + lin_mat_pes][c - parameters.raio + col_mat_pes] - r * 1000000) / 1000;;
					b = cudaOldimage[l - parameters.raio + lin_mat_pes][c - parameters.raio + col_mat_pes] - r * 1000000 - g * 1000;
					acumular += (r * cudaMatrizPesos[lin_mat_pes][col_mat_pes]);
					acumulag += (g * cudaMatrizPesos[lin_mat_pes][col_mat_pes]);
					acumulab += (b * cudaMatrizPesos[lin_mat_pes][col_mat_pes]);
				}
			}
			nr = acumular;
			ng = acumulag;
			nb = acumulab;
			cudaNewimage[l][c] = nr * 1000000 + ng * 1000 + nb;
		}
	}

}

int main() {
	FILE *arqin;
	FILE *arqout;
	char narqin[MAX_NAME] = "c:\\temp\\reddead.ppm";
	char narqout[MAX_NAME] = "c:\\temp\\reddead2.ppm";
	char key[128];
	int i, j, max, r, g, b; //auxiliares
	int Blocks = 1;
	int ThreadsPerBlock = 1;

	struct param parameters;
	cudaError_t cudaStatus;

	parameters.totalthreads = Blocks * ThreadsPerBlock;

	printf("Qual raio?\n");
	scanf("%d", &parameters.raio);

	printf("Arquivo de entrada: %s\n", narqin);
	arqin = fopen(narqin, "r");

	if (arqin == NULL) {
		printf("Erro na abertura do arquivo %s\n", narqin);
		return 1;
	}

	printf("Arquivo de saida: %s\n", narqout);
	arqout = fopen(narqout, "w");

	if (arqout == NULL) {
		printf("Erro na abertura do arquivo %s\n", narqin);
		return 1;
	}

	fscanf(arqin, "%s", key);//leio cabeçalho
	fprintf(arqout, "%s\n", key);//já escrevo o cabeçalho no novo arquivo
	printf("Arquivo tipo: %s \n", key);
	fscanf(arqin, "%d %d %d", &parameters.colunas, &parameters.linhas, &max);//leio mais dados do cabeçalho
	fprintf(arqout, "%d %d \n%d", parameters.colunas, parameters.linhas, max);//já escrevo esses dados no novo arquivo
	printf("Colunas = %d \nLinhas = %d \n", parameters.colunas, parameters.linhas);

	//vamos definir o tamanho para cada um
	parameters.tamanho = parameters.linhas / parameters.totalthreads;
	if ((parameters.linhas % parameters.totalthreads) > 0){
		parameters.sobra = parameters.linhas % parameters.totalthreads;
	}
	else {
		parameters.sobra = 0;
	}

	printf("Tamanho %d\n", parameters.tamanho);

	//por enquanto nao vamos aceitar imagem com apenas uma linha
	if (parameters.linhas < parameters.totalthreads){
		printf("Mais threads do que dados %s\n", narqin);
		return 0;
	}
	
	int **oldimage = AlocaMatriz(parameters.linhas, parameters.colunas);
	int **newimage = AlocaMatriz(parameters.linhas, parameters.colunas);
	matrizpesos = AlocaMatrizDouble(parameters.raio * 2 + 1, parameters.raio * 2 + 1);
	InicializaMatrizPesos(parameters.raio);
	
	for (i = 0; i <= parameters.linhas - 1; i++)
		for (j = 0; j <= parameters.colunas - 1; j++) {
		fscanf(arqin, " %d %d %d ", &r, &g, &b);
		//printf("RGB: %d %d %d \n", r, g, b);
		oldimage[i][j] = r * 1000000 + g * 1000 + b;
		/*
		rgb = oldimage[i][j];
		nr = rgb/1000000;
		ng = (rgb-r*1000000)/1000;
		nb = rgb-r*1000000-g*1000;
		if ((nr != r) || (ng != g) || (nb != b)) printf("errooou");
		printf("Valor: %d\n", rgb);
		printf("Valor R: %d\n", r);
		printf("Valor G: %d\n", g);
		printf("Valor B: %d\n", b);
		*/
		}

	// aloca a memória no device
	int size_m_int = parameters.linhas*parameters.colunas*sizeof(int);	// tamanho da memória que será aolocado para as matrizes
	int size_m_double = (parameters.raio * 2 + 1) * (parameters.raio * 2 + 1) * sizeof(double);	// tamanho da memória que será aolocado para a matriz de pesos
	int **doldimage, **dnewimage;
	double **dmatrizpesos;

	printf("1...alocando doldimage na GPU...");
	cudaStatus = cudaMalloc((void**)&doldimage, size_m_int);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return 1;
	}
	printf("2...alocando dnewimage na GPU...");
	cudaStatus = cudaMalloc((void**)&dnewimage, size_m_int);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return 1;
	}
	printf("3...alocando dmatrizpesos na GPU...");
	cudaStatus = cudaMalloc((void**)&dmatrizpesos, size_m_double);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return 1;
	}

	// copia as matrizes do host para o device
	printf("4...copiando oldimage para GPU...");
	cudaStatus = cudaMemcpy(doldimage, oldimage, size_m_int, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		return 1;
	}
	printf("5...copiando matrizpesos para GPU...");
	cudaStatus = cudaMemcpy(dmatrizpesos, matrizpesos, size_m_double, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		return 1;
	}

	//executa o kernel
	printf("6...rodando kernel...");
	filtra<<<Blocks, ThreadsPerBlock>>>(doldimage, dnewimage, dmatrizpesos, parameters);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	printf("7...sincronizando com device...");
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return 1;
	}

	//copia matriz resultante da GPU para a CPU
	printf("8...copiando dnewimage da GPU para processador...");
	cudaStatus = cudaMemcpy(newimage, dnewimage, size_m_int, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		return 1;
	}

	//escrever novo arquivo    
	printf("9...escrevendo nova imagem...");
	for (i = 0; i <= parameters.linhas - 1; i++){
		fprintf(arqout, "\n");
		for (j = 0; j <= parameters.colunas - 1; j++) {
			r = newimage[i][j] / 1000000;
			g = (newimage[i][j] - r * 1000000) / 1000;
			b = newimage[i][j] - r * 1000000 - g * 1000;
			fprintf(arqout, "%d %d %d ", r, g, b);
		}
	}

	//    for (i = 0; i <= linhas - 1; i++) for (j = 0; j <= colunas - 1; j++) printf("RGB: %d %d %d \n", newimage[i][j*3], newimage[i][j*3+1], newimage[i][j*3+2]);

	printf("10...liberando matrizes...");
	LiberaMatriz(parameters.linhas, parameters.colunas, oldimage);
	LiberaMatriz(parameters.linhas, parameters.colunas, newimage);
	LiberaMatrizDouble(parameters.raio * 2 + 1, parameters.raio * 2 + 1, matrizpesos);
	cudaFree(doldimage);
	cudaFree(dnewimage);
	cudaFree(dmatrizpesos);

	fclose(arqin);
	fclose(arqout);

	printf("Fim programa.\n");
	return 0;
}

void InicializaMatrizPesos(int raio){
	int i, j;
	double e, g;
	double somapesos = 0;
	float sigma = raio;
	for (i = 0; i < sigma * 2 + 1; i++){
		//printf("\n");
		for (j = 0; j < raio * 2 + 1; j++){
			e = pow((float)exp(1.0), ((-1)*(pow((i - sigma), 2) + pow((j - sigma), 2)) / (2 * pow(sigma, 2))));
			//printf("P(%d,%d)\n", i, j);
			//printf("E = %.4f - PARTEDECIMA = %.4f\n", e, partedecima);
			g = e / (2 * PI*pow(sigma, 2));
			matrizpesos[i][j] = g;
			somapesos += g;
			//printf("P(%d,%d) = %.4f ;", i, j, g);
		}
	}
	for (i = 0; i < sigma * 2 + 1; i++){
		//printf("de novo \n");
		for (j = 0; j < raio * 2 + 1; j++){
			matrizpesos[i][j] = matrizpesos[i][j] / somapesos;
			//printf("P(%d,%d) = %.4f ;", i, j, matrizpesos[i][j]);
		}
	}
	//printf("somapesos = %.5f\n", somapesos);
}

int **AlocaMatriz(int lin, int col){
	int **mat;  /* ponteiro para a matriz */
	int i;    /* variavel auxiliar      */
	if (lin < 1 || col < 1) { /* verifica parametros recebidos */
		printf("** Erro: Parametro invalido **\n");
		return(NULL);
	}
	/* aloca as linhas da matriz */
	mat = (int **)calloc(lin, sizeof(int *));
	if (mat == NULL) {
		printf("** Erro: Memoria Insuficiente **");
		return(NULL);
	}
	/* aloca as colunas da matriz */
	for (i = 0; i < lin; i++){
		mat[i] = (int*)calloc(col, sizeof(int));
		if (mat[i] == NULL) {
			printf("** Erro: Memoria Insuficiente **");
			return(NULL);
		}
	}
	return(mat); /* retorna o ponteiro para a matriz */
}

int **LiberaMatriz(int lin, int col, int **mat){
	int i;  /* variavel auxiliar */
	if (mat == NULL) return(NULL);
	if (lin < 1 || col < 1){  /* verifica parametros recebidos */
		printf("** Erro: Parametro invalido **\n");
		return(mat);
	}
	for (i = 0; i<lin; i++) free(mat[i]); /* libera as linhas da matriz */
	free(mat);      /* libera a matriz */
	return(NULL); /* retorna um ponteiro nulo */
}

double **AlocaMatrizDouble(int lin, int col){
	double **mat;  /* ponteiro para a matriz */
	int i;    /* variavel auxiliar      */
	if (lin < 1 || col < 1) { /* verifica parametros recebidos */
		printf("** Erro: Parametro invalido **\n");
		return(NULL);
	}
	/* aloca as linhas da matriz */
	mat = (double **)calloc(lin, sizeof(double *));
	if (mat == NULL) {
		printf("** Erro: Memoria Insuficiente **");
		return(NULL);
	}
	/* aloca as colunas da matriz */
	for (i = 0; i < lin; i++){
		mat[i] = (double*)calloc(col, sizeof(double));
		if (mat[i] == NULL) {
			printf("** Erro: Memoria Insuficiente **");
			return(NULL);
		}
	}
	return(mat); /* retorna o ponteiro para a matriz */
}

double **LiberaMatrizDouble(int lin, int col, double **mat){
	int i;  /* variavel auxiliar */
	if (mat == NULL) return(NULL);
	if (lin < 1 || col < 1){  /* verifica parametros recebidos */
		printf("** Erro: Parametro invalido **\n");
		return(mat);
	}
	for (i = 0; i<lin; i++) free(mat[i]); /* libera as linhas da matriz */
	free(mat);      /* libera a matriz */
	return(NULL); /* retorna um ponteiro nulo */
}