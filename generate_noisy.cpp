#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <vector>
#include <iostream>
#include <format>
#include <unordered_map>
#include <numeric>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <random>


std::vector<int> getRandomNumbers(int N) {
    int min = 0;
    int max = 64 * 64;
    std::random_device rd;  // Источник энтропии
    std::mt19937 gen(rd()); // Генератор с начальным зерном
    std::uniform_int_distribution<> dist(min, max); // Равномерное распределение

    std::vector<int> numbers;
    for (int i = 0; i < N; ++i) {
        numbers.push_back(dist(gen));
    }
    return numbers;
}


std::vector<int>
readImageForLearn(const std::vector<unsigned char *> &imagesLink, int width, int height) {
    std::vector<int> resultImage(width * height, 0);
    for (auto imagePtr: imagesLink) {
        std::vector<unsigned char> learnByImage(imagePtr, imagePtr + width * height);

        for (int i = 0; i < learnByImage.size(); i += 1) {
            if (learnByImage[i] == 255) {
                resultImage[i] = 1;
            } else {
                resultImage[i] = -1;
            }
        }
    }
    return resultImage;
}

void save(int noisyLevel, std::vector<int> image) {
    int countOfReplace = 64 * 64 * ((double) noisyLevel / 100);
    std::vector<int> bitToReplace = getRandomNumbers(countOfReplace);
    for (int i = 0; i < countOfReplace; i++) {
        image[bitToReplace[i]] = image[bitToReplace[i]] == 1 ? -1 : 1;
    }

    std::vector<unsigned char> vectorToPrint(64 * 64, 0);
    for (int i = 0; i < image.size(); i++) {
        vectorToPrint[i] = image[i] == 1 ? 255 : 0;
    }

    std::string filename = std::format("/Users/il.d.zhukov/CLionProjects/HopfieldNN/slim_digit/noisy/1/noisy_{}.png", noisyLevel);
    stbi_write_png(filename.data(), 64, 64, 1, vectorToPrint.data(), 64);
}

int main() {
    int width, height, channels;

    unsigned char *learn0 = stbi_load("/Users/il.d.zhukov/CLionProjects/HopfieldNN/slim_digit/learn_1.png",
                                      &width, &height, &channels, 0);


    std::vector<unsigned char *> learnLinks{learn0};
    std::vector<int> image = readImageForLearn(learnLinks, width, height);


    save(10, readImageForLearn(learnLinks, width, height));
    save(20, readImageForLearn(learnLinks, width, height));
    save(30, readImageForLearn(learnLinks, width, height));
    save(40, readImageForLearn(learnLinks, width, height));
    save(50, readImageForLearn(learnLinks, width, height));
    save(60, readImageForLearn(learnLinks, width, height));
    save(70, readImageForLearn(learnLinks, width, height));

    return 0;
}