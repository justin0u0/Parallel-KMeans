CXX = g++
LDFLAGS = -lpng
TARGETS = kmeans_sequential kmeans_omp kmeans_omp2 kmeans_cuda

kmeans_omp: CXXFLAGS += -fopenmp
kmeans_omp2: CXXFLAGS += -fopenmp -std=c++11

NVCC = nvcc
NVCCFLAGS = -O3 -std=c++11 -Xptxas=-v -arch=sm_61

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)

kmeans_cuda: kmeans_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<
