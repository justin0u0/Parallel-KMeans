#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>
#include "kmeans_png.h"

// K-Means point holding a single image channel
struct Point {
	uint8_t x;   // R
	uint8_t y;   // G
	uint8_t z;   // B
	int cluster; // the cluster id
};

// K-Means cluster centeral points
struct Cluster {
	uint8_t x;    // R
	uint8_t y;    // G
	uint8_t z;    // B
};

__global__ void clustering(int n, int k, Point* points, Cluster* clusters) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n) return;

	int min_dist = 0x7fffffff;
	int cluster;

	for (int i = 0; i < k; ++i) {
		int dist = (
			(points[idx].x - clusters[i].x) * (points[idx].x - clusters[i].x) + 
			(points[idx].y - clusters[i].y) * (points[idx].y - clusters[i].y) + 
			(points[idx].z - clusters[i].z) * (points[idx].z - clusters[i].z)
		);

		if (min_dist > dist) {
			min_dist = dist;
			cluster = i;
		}
	}

	points[idx].cluster = cluster;
}

__global__ void recentering_phase1(int n, int cluster, Point* points, int* sumX, int* sumY, int* sumZ, int* count) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n) return;

	if (points[idx].cluster == cluster) {
		atomicAdd(sumX, points[idx].x);
		atomicAdd(sumY, points[idx].y);
		atomicAdd(sumZ, points[idx].z);
		atomicAdd(count, 1);
	}
}

__global__ void recentering_phase2(Cluster* cluster, int* sumX, int* sumY, int* sumZ, int* count, bool* done) {
	if ((*count) > 0) {
		Cluster old = *cluster;

		cluster->x = (*sumX) / (*count);
		cluster->y = (*sumY) / (*count);
		cluster->z = (*sumZ) / (*count);

		*done = (old.x == cluster->x && old.y == cluster->y && old.z == cluster->z);
	}
}

int main(int argc, const char** argv) {
	const char* infile = argv[1];
	const char* outfile = argv[2];
	const int k = atoi(argv[3]);

	unsigned char* image;
	unsigned int height;
	unsigned int width;
	unsigned int channels;

	int err = read_png(infile, &image, height, width, channels);
	if (err != 0) {
		printf("fail to read input png file\n");
		exit(err);
	}

	int n = height * width;
	Point* points = (Point*)malloc(n * sizeof(Point));
	for (int i = 0; i < n; ++i) {
		points[i].x = image[i * channels];
		points[i].y = image[i * channels + 1];
		points[i].z = image[i * channels + 2];
		points[i].cluster = -1;
	}

	Cluster* clusters = (Cluster*)malloc(k * sizeof(Cluster));
	std::mt19937 mt{std::random_device{}()};
	mt.seed(5);
	std::uniform_int_distribution<> dist(0, n - 1);
	for (int i = 0; i < k; ++i) {
		int j = dist(mt);
		clusters[i].x = points[j].x;
		clusters[i].y = points[j].y;
		clusters[i].z = points[j].z;
	}

	const int threads_per_block = 1024;
	const int number_of_blocks = (n + threads_per_block - 1) / threads_per_block;

	// allocate device memory
	Point* dpoints;
	Cluster* dclusters;

	cudaMalloc((void**)&dpoints, n * sizeof(Point));
	cudaMalloc((void**)&dclusters, k * sizeof(Cluster));

	cudaMemcpy(dpoints, points, n * sizeof(Point), cudaMemcpyHostToDevice);
	cudaMemcpy(dclusters, clusters, k * sizeof(Cluster), cudaMemcpyHostToDevice);
	
	int* dall; // {sumX,sumY,sumZ,count}
	bool* ddone;
	cudaMalloc((void**)&dall, 4 * sizeof(int));
	cudaMalloc((void**)&ddone, sizeof(bool));

	while (true) {
		clustering<<<number_of_blocks, threads_per_block>>>(n, k, dpoints, dclusters);

		bool alldone = true;
		for (int i = 0; i < k; ++i) {
			cudaMemset(dall, 0, 4 * sizeof(int));

			bool done = true;
			cudaMemcpy(ddone, &done, sizeof(bool), cudaMemcpyHostToDevice);

			recentering_phase1<<<number_of_blocks, threads_per_block>>>(n, i, dpoints, &dall[0], &dall[1], &dall[2], &dall[3]);
			recentering_phase2<<<1, 1>>>(&dclusters[i], &dall[0], &dall[1], &dall[2], &dall[3], ddone);

			cudaMemcpy(&done, ddone, sizeof(bool), cudaMemcpyDeviceToHost);

			alldone &= done;
		}

		if (alldone) break;
	}

	// copy device memory back to host
	cudaMemcpy(points, dpoints, n * sizeof(Point), cudaMemcpyDeviceToHost);
	cudaMemcpy(clusters, dclusters, k * sizeof(Cluster), cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(dpoints);
	cudaFree(dclusters);
	cudaFree(dall);

	for (int i = 0; i < n; ++i) {
		Cluster* cluster = &clusters[points[i].cluster];
		image[i * channels] = cluster->x;
		image[i * channels + 1] = cluster->y;
		image[i * channels + 2] = cluster->z;
	}

	err = write_png(outfile, image, height, width, channels);
	if (err != 0) {
		printf("fail to write output png file\n");
		exit(err);
	}

	/*
	cudaError_t cudaErr = cudaGetLastError();
	if (cudaErr != cudaSuccess) {
		printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr));
		exit(1);
	}
	*/

	delete[] clusters;
	delete[] points;

	return 0;
}
