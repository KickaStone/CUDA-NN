//
// Created by JunchengJi on 7/30/2023.
//

#ifndef NETWORK_MNIST_LOADER_H
#define NETWORK_MNIST_LOADER_H

#include <iostream>
#include <fstream>
#include <vector>

int reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

void load_mnist(const char *image_file, const char *label_file, std::vector<double *> &data, std::vector<int> &label) {
    FILE *image_fp = fopen(image_file, "rb");
    if (!image_fp) {
        std::cout << "Cannot open image file " << image_file << std::endl;
        exit(1);
    }

    int magic_number = 0;
    int n_images = 0;
    int img_width = 0;
    int img_height = 0;
    fread(&magic_number, sizeof(magic_number), 1, image_fp);
    magic_number = reverseInt(magic_number);
    fread(&n_images, sizeof(n_images), 1, image_fp);
    n_images = reverseInt(n_images);
    fread(&img_height, sizeof(img_height), 1, image_fp);
    img_height = reverseInt(img_height);
    fread(&img_width, sizeof(img_width), 1, image_fp);
    img_width = reverseInt(img_width);

    for (int i = 0; i < n_images; i++) {
        double *img = new double[img_height * img_width];
        for (int j = 0; j < img_height * img_width; ++j) {
            unsigned char temp = 0;
            fread(&temp, sizeof(temp), 1, image_fp);
            img[j] = (double) temp / 255.0;
        }
        data.push_back(img);
    }

    fclose(image_fp);

    FILE *label_fp = fopen(label_file, "rb");
    if (!label_fp) {
        std::cout << "Cannot open label file " << label_file << std::endl;
        exit(1);
    }

    fread(&magic_number, sizeof(magic_number), 1, label_fp);
    magic_number = reverseInt(magic_number);
    fread(&n_images, sizeof(n_images), 1, label_fp);
    n_images = reverseInt(n_images);

    for (int i = 0; i < n_images; i++) {
        unsigned char temp = 0;
        fread(&temp, sizeof(temp), 1, label_fp);
        label.push_back((int) temp);
    }
    fclose(label_fp);
}

#endif //NETWORK_MNIST_LOADER_H
