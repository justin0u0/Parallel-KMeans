#ifndef _KMEANS_PNG_H_
#define _KMEANS_PNG_H_

#include <png.h>

#define PNG_SIG_SIZE 8

int read_png(const char* filename, unsigned char** image, unsigned int& height, unsigned int& width, unsigned int& channels) {
	unsigned char sig[PNG_SIG_SIZE];

	FILE* fp = fopen(filename, "rb");
	if (!fp) {
		return 1;
	}

	fread(sig, 1, PNG_SIG_SIZE, fp);
	if (!png_check_sig(sig, PNG_SIG_SIZE)) {
		return 1;
	}

	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png_ptr) {
		return 1;
	}

	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) {
		png_destroy_read_struct(&png_ptr, NULL, NULL);
		return 1;
	}

	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, PNG_SIG_SIZE);
	png_read_info(png_ptr, info_ptr);

 	width = png_get_image_width(png_ptr, info_ptr);
  height = png_get_image_height(png_ptr, info_ptr);
  png_uint_32 color_type = png_get_color_type(png_ptr, info_ptr);
  png_uint_32 bit_depth = png_get_bit_depth(png_ptr, info_ptr);

	if (color_type != PNG_COLOR_TYPE_RGB) {
		// our k-means implementation only support png image with color type RGB (3 channels)
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		return 2;
	}

	if (bit_depth != 8) {
		// our k-means implementation only support png image with bit depth 8
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		return 2;
	}

	channels = png_get_channels(png_ptr, info_ptr);
	if (channels != 3) {
		// our k-means implementation only support png image with channels equals 3
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		return 2;
	}

	// number of bytes needed to hold a transformed row,
	// equals to `width * channels * bit_depth / 8`
	// should also equals to png_get_rowbytes(png_ptr, info_ptr)
	size_t row_bytes = width * channels;

	*image = (unsigned char*)malloc(height * row_bytes);
	if ((*image) == NULL) {
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		return 3;
	}

	png_bytep* image_ptr = (png_bytep*)malloc(height * sizeof(png_bytep));
	for (unsigned int i = 0; i < height; ++i) {
		image_ptr[i] = (*image) + i * row_bytes;
	}

	png_read_image(png_ptr, image_ptr);
	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

	free(image_ptr);
	fclose(fp);
	return 0;
}

int write_png(const char* filename, unsigned char* image, unsigned int height, unsigned int width, unsigned int channels) {
	FILE* fp = fopen(filename, "wb");

	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png_ptr) {
		return 1;
	}

	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) {
		png_destroy_write_struct(&png_ptr, NULL);
		return 1;
	}

	png_init_io(png_ptr, fp);
	png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
	png_write_info(png_ptr, info_ptr);
	png_set_compression_level(png_ptr, 1);

	png_bytep* image_ptr = (png_bytep*)malloc(height * sizeof(png_bytep));
	size_t row_bytes = width * channels;

	for (int i = 0; i < height; ++i) {
		image_ptr[i] = image + i * row_bytes;
	}

	png_write_image(png_ptr, image_ptr);
	png_write_end(png_ptr, NULL);
	png_destroy_write_struct(&png_ptr, &info_ptr);

	fclose(fp);
	return 0;
}

#endif // _KMEANS_PNG_H_
