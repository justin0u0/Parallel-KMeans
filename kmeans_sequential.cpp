#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
#include "kmeans.h"
#include "kmeans_png.h"

void clustering(int n, int k, Point* points, Point* centers) {
	for (int i = 0; i < n; ++i) {
		int min_dist = 0x7fffffff;
		int cluster;

		for (int j = 0; j < k; ++j) {
			int dist = (
				(points[i].x - centers[j].x) * (points[i].x - centers[j].x) + 
				(points[i].y - centers[j].y) * (points[i].y - centers[j].y) + 
				(points[i].z - centers[j].z) * (points[i].z - centers[j].z)
			);

			if (min_dist > dist) {
				min_dist = dist;
				cluster = j;
			}
		}

		points[i].cluster = cluster;
	}
}

bool recentering(int n, int k, Point* points, Point* centers) {
	int sumX[k] = {0};
	int sumY[k] = {0};
	int sumZ[k] = {0};
	int count[k] = {0};

	for (int i = 0; i < n; ++i) {
		int cluster = points[i].cluster;
		++count[cluster];
		sumX[cluster] += points[i].x;
		sumY[cluster] += points[i].y;
		sumZ[cluster] += points[i].z;
	}

	bool done = true;
	for (int i = 0; i < k; ++i) {
		if (count[i] > 0) {
			Point old = centers[i];

			centers[i].x = sumX[i] / count[i];
			centers[i].y = sumY[i] / count[i];
			centers[i].z = sumZ[i] / count[i];

			if (old.x != centers[i].x || old.y != centers[i].y || old.z != centers[i].z) {
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

	Point* centers = (Point*)malloc(k * sizeof(Point));
	std::mt19937 mt{std::random_device{}()};
	std::uniform_int_distribution<> dist(0, n - 1);
	for (int i = 0; i < k; ++i) {
		centers[i] = points[dist(mt)];
	}

	do {
		clustering(n, k, points, centers);
	} while (!recentering(n, k, points, centers));

	for (int i = 0; i < n; ++i) {
		Point* center = &centers[points[i].cluster];
		image[i * channels] = center->x;
		image[i * channels + 1] = center->y;
		image[i * channels + 2] = center->z;
	}

	err = write_png(outfile, image, height, width, channels);
	if (err != 0) {
		printf("fail to write output png file\n");
		exit(err);
	}

	return 0;
}
