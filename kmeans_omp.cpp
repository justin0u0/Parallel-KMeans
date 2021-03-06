#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
#include <omp.h>
#include "kmeans.h"
#include "kmeans_png.h"

void clustering(int n, int k, Point* points, Cluster* clusters) {
	for (int i = 0; i < k; i++) {
		clusters[i].size = 0;
	}

	#pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		int min_dist = 0x7fffffff;
		int cluster;

		for (int j = 0; j < k; ++j) {
			int dist = (
				(points[i].x - clusters[j].x) * (points[i].x - clusters[j].x) + 
				(points[i].y - clusters[j].y) * (points[i].y - clusters[j].y) + 
				(points[i].z - clusters[j].z) * (points[i].z - clusters[j].z)
			);

			if (min_dist > dist) {
				min_dist = dist;
				cluster = j;
			}
		}

		points[i].cluster = cluster;

		#pragma omp critical
		clusters[cluster].points[clusters[cluster].size++] = i;
	}
}

bool recentering(int n, int k, Point* points, Cluster* clusters) {
	bool done = true;

	for (int i = 0; i < k; ++i) {
		Cluster* cluster = &clusters[i];

		int sumX = 0;
		int sumY = 0;
		int sumZ = 0;

		#pragma omp parallel for reduction(+:sumX, sumY, sumZ)
		for (int j = 0; j < cluster->size; ++j) {
			Point* point = &points[cluster->points[j]];
			sumX += point->x;
			sumY += point->y;
			sumZ += point->z;
		}

		if (cluster->size > 0) {
			Cluster old = clusters[i];
			Cluster *cur = &clusters[i];

			cur->x = sumX / cluster->size;
			cur->y = sumY / cluster->size;
			cur->z = sumZ / cluster->size;

			if (old.x != cur->x || old.y != cur->y || old.z != cur->z) {
				done = false;
			}
		}
	}

	return done;
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
	std::uniform_int_distribution<> dist(0, n - 1);
	for (int i = 0; i < k; ++i) {
		int j = dist(mt);
		clusters[i].x = points[j].x;
		clusters[i].y = points[j].y;
		clusters[i].z = points[j].z;
		clusters[i].size = 0;
		clusters[i].points = (int*)malloc(n * sizeof(int));
	}

	do {
		clustering(n, k, points, clusters);
	} while (!recentering(n, k, points, clusters));

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

	return 0;
}
