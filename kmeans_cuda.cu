#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>
#include "kmeans_png.h"

__inline__ __device__ uint4 warp_reduce_sum(uint4 val) {
	// we ensure that all threads size > warp size so just use full mask directly

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val.x += __shfl_down_sync(0xffffffff, val.x, offset);
    val.y += __shfl_down_sync(0xffffffff, val.y, offset);
    val.z += __shfl_down_sync(0xffffffff, val.z, offset);
		val.w += __shfl_down_sync(0xffffffff, val.w, offset);
	}

  return val;
}

__global__ void clustering(int n, int k, uint3* __restrict__ points, int* __restrict__ clusters, const uint3* __restrict__ centers) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n) return;

	// copy global memory to local
	uint3 point = points[idx];
	int min_dist = 0x7fffffff;
	int cluster;

	for (int i = 0; i < k; ++i) {
		int dist = (
			(point.x - centers[i].x) * (point.x - centers[i].x) +
			(point.y - centers[i].y) * (point.y - centers[i].y) +
			(point.z - centers[i].z) * (point.z - centers[i].z)
		);

		if (min_dist > dist) {
			min_dist = dist;
			cluster = i;
		}
	}

	clusters[idx] = cluster;
}

__global__ void recentering_phase1(int n, int cluster, const uint3* __restrict__ points, const int* __restrict__ clusters, uint4* __restrict__ all) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	uint4 sum = make_uint4(0, 0, 0, 0);
	if (idx < n && cluster == clusters[idx]) {
		uint3 point = points[idx];
		sum = make_uint4(point.x, point.y, point.z, 1);
	}

	sum = warp_reduce_sum(sum);

	if ((threadIdx.x & (warpSize - 1)) == 0) {
		atomicAdd(&all->x, sum.x);
		atomicAdd(&all->y, sum.y);
		atomicAdd(&all->z, sum.z);
		atomicAdd(&all->w, sum.w);
	}
}

__global__ void recentering_phase2(uint3* center, uint4* all, bool* done) {
	if (all->w > 0) {
		uint3 old = *center;

		center->x = (all->x) / (all->w);
		center->y = (all->y) / (all->w);
		center->z = (all->z) / (all->w);

		*done = (old.x == center->x && old.y == center->y && old.z == center->z);
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

	uint3* points = (uint3*)malloc(n * sizeof(uint3)); // {R,G,B}
	int* clusters = (int*)malloc(n * sizeof(int));     // list of cluster id
	for (int i = 0; i < n; ++i) {
		points[i].x = image[i * channels];
		points[i].y = image[i * channels + 1];
		points[i].z = image[i * channels + 2];
		clusters[i] = -1;
	}

	uint3* centers = (uint3*)malloc(k * sizeof(uint3)); // center of clusters {R,G,B}
	std::mt19937 mt{std::random_device{}()};
	mt.seed(5);
	std::uniform_int_distribution<> dist(0, n - 1);
	for (int i = 0; i < k; ++i) {
		centers[i] = points[dist(mt)];
	}

	const int threads_per_block = 1024;
	const int number_of_blocks = (n + threads_per_block - 1) / threads_per_block;

	// allocate device memory
	uint3* dpoints;
	int* dclusters;
	uint3* dcenters;

	cudaMalloc((void**)&dpoints, n * sizeof(uint3));
	cudaMalloc((void**)&dclusters, n * sizeof(int));
	cudaMalloc((void**)&dcenters, k * sizeof(uint3));

	cudaMemcpy(dpoints, points, n * sizeof(uint3), cudaMemcpyHostToDevice);
	cudaMemcpy(dclusters, clusters, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dcenters, centers, k * sizeof(uint3), cudaMemcpyHostToDevice);

	uint4* dall; // {sumX,sumY,sumZ,count}
	bool* ddone;
	cudaMalloc((void**)&dall, sizeof(uint4));
	cudaMalloc((void**)&ddone, sizeof(bool));

	while (true) {
		clustering<<<number_of_blocks, threads_per_block>>>(n, k, dpoints, dclusters, dcenters);

		bool alldone = true;
		for (int i = 0; i < k; ++i) {
			cudaMemset(dall, 0, sizeof(uint4));

			bool done = true;
			cudaMemcpy(ddone, &done, sizeof(bool), cudaMemcpyHostToDevice);

			recentering_phase1<<<number_of_blocks, threads_per_block>>>(n, i, dpoints, dclusters, dall);
			recentering_phase2<<<1, 1>>>(&dcenters[i], dall, ddone);

			cudaMemcpy(&done, ddone, sizeof(bool), cudaMemcpyDeviceToHost);

			alldone &= done;
		}

		if (alldone) break;
	}

	// copy device memory back to host
	// cudaMemcpy(points, dpoints, n * sizeof(uint3), cudaMemcpyDeviceToHost);
	cudaMemcpy(clusters, dclusters, n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(centers, dcenters, k * sizeof(uint3), cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(dpoints);
	cudaFree(dclusters);
	cudaFree(dcenters);
	cudaFree(dall);

	for (int i = 0; i < n; ++i) {
		uint3* center = &centers[clusters[i]];
		image[i * channels] = center->x;
		image[i * channels + 1] = center->y;
		image[i * channels + 2] = center->z;
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
