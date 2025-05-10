#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <vector>
#include <iostream>
#include <format>
#include <unordered_map>
#include <numeric>
#include <algorithm>

#define EPOCH 10

double normFun(std::vector<double> vector) {
    double sum = 0;
    for (const auto &item: vector) {
        sum += item * item;
    }
    return sqrt(sum);
}

void copy(std::vector<int> source, std::vector<double>& target){
    for (int i = 0; i < target.size(); i++) {
        target[i] = (double) source[i];
    }
}

double sсalarMultiplication(std::vector<double> v1, const std::vector<int>& v2) {
    double res = 0;
    for (int i = 0; i < v1.size(); i++) {
        res += v1[i] * (double) v2[i];
    }
    return res;
}

std::vector<double> getMinusVector(std::vector<double>& v1, const std::vector<int>& v2) {
    double multi = sсalarMultiplication(v1, v2);
    for (int i = 0; i < v1.size(); i++) {
        v1[i] = v1[i] * multi;
    }
    return v1;
}

std::vector<std::vector<double>> trainWithNorm(std::vector<std::vector<int>>& images) {
    std::vector<std::vector<double>> ort_vectors;

    for (const auto &item: images) {
        std::vector<double> v_ort(item.size());
        copy(item, v_ort);

        for (int i = 0; i < ort_vectors.size(); i++) {
            std::vector<double> v_ort_local = ort_vectors[i];

            std::vector<double> minusVector = getMinusVector(v_ort_local, item);
            double sum4 = 0;
            for (int j = 0; j < minusVector.size(); j++) {
                sum4 += minusVector[j];
            }
            for (int j = 0; j < v_ort_local.size(); j++) {
                v_ort[j] -= minusVector[j];
            }

        }
        double norm = normFun(v_ort);
        if (norm > 0.000000001) {
            for (int i = 0; i < v_ort.size(); i++) {
                v_ort[i] /= norm;
            }
            ort_vectors.push_back(v_ort);
        }
    }
    int size = images[0].size();

    std::vector<std::vector<double>> matrix;
    for (int i = 0; i < size; ++i) {
        std::vector<double> row(size, 0);
        matrix.push_back(row);
    }


    for (const auto &item: ort_vectors) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i == j) {
                    matrix[i][j] = 0;
                } else {
                    matrix[i][j] += item[i] * item[j];
                }
            }
        }
    }

    std::cout << "matrix" << std::endl;
    double sumRes = 0;
    std::unordered_map<double, int> digits;
    for (int iy = 0; iy < size; iy++) {
        for (int ix = 0; ix < size; ix++) {
            sumRes += matrix[iy][ix];
            double value = matrix[iy][ix];
            int digitCount = digits[value];
            digits[value] = digitCount + 1;
        }
    }
    for (const auto &item: digits) {
        //std::cout << item.first << " " << item.second << std::endl;
    }

    std::vector<std::pair<double, int>> vec(digits.begin(), digits.end());

    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });


    for (const auto& pair : vec) {
        //std::cout << pair.second << ": " << pair.first << '\n';
    }

    return matrix;
}

void recognize(std::vector<int> inputImage, std::vector<std::vector<double>> weight, int testWidth, int testHeight) {
    size_t size = inputImage.size();
    std::vector<double> sVector;
    int prev = std::accumulate(inputImage.begin(), inputImage.end(), 0);
    for (int i = 0; i < EPOCH; i++) {
        std::vector<int> old = inputImage;
        for (int neuron = 0; neuron < size; neuron++) {
            double s = 0;
            for (int j = 0; j < size; j++) {
                s += weight[neuron][j] * (double) old[j];
            }
            //std::cout << "neuron " << neuron << " " << s << std::endl;
            sVector.push_back(s);
            if (s > 0.0) {
                inputImage[neuron] = 1;
            } else if (s == 0) {
                inputImage[neuron] = 0;
            } else {
                inputImage[neuron] = -1;
            }
        }
        int current = std::accumulate(inputImage.begin(), inputImage.end(), 0);

        std::vector<unsigned char> vectorToPrint(size, 0);
        for (int j = 0; j < size; j++) {
            if (inputImage[j] == 1) {
                vectorToPrint[j] = 255;
            } else if (inputImage[j] == 0) {
                vectorToPrint[j] = 255 / 2;
            } else {
                vectorToPrint[j] = 0;
            }
        }
        std::cout << "epoch " << i << " finish" << std::endl;
        std::string filename = std::format("/Users/il.d.zhukov/CLionProjects/HopfieldNN/output/output_{}.png", i);
        stbi_write_png(filename.data(), testWidth, testHeight, 1, vectorToPrint.data(), testWidth);

        if (current == prev) {
            std::cout << "learn finish" << std::endl;
            break;
        }
        prev = current;
    }
}

std::vector<std::vector<int>>
readImageForLearn(const std::vector<unsigned char *> &imagesLink, int width, int height) {
    std::vector<std::vector<int>> result;
    for (auto imagePtr: imagesLink) {
        std::vector<unsigned char> learnByImage(imagePtr, imagePtr + width * height);
        std::vector<int> resultImage(width * height, 0);

        for (int i = 0; i < learnByImage.size(); i += 1) {
            if (learnByImage[i] == 255) {
                resultImage[i] = 1;
            } else {
                resultImage[i] = -1;
            }
        }
        result.push_back(resultImage);
    }
    return result;
}

std::vector<std::vector<int>>
readImageForTest(const std::vector<unsigned char *> &imagesLink, int width, int height) {
    std::vector<std::vector<int>> result;
    for (auto imagePtr: imagesLink) {
        std::vector<unsigned char> learnByImage(imagePtr, imagePtr + width * height);
        std::vector<int> resultImage(width * height, 0);

        for (int i = 0; i < learnByImage.size(); i += 1) {
            if (learnByImage[i] == 255) {
                resultImage[i] = 1;
            } else {
                resultImage[i] = 0;
            }
        }
        result.push_back(resultImage);
    }
    return result;
}

int main() {
    int width, height, channels;

    unsigned char *learn0 = stbi_load("/Users/il.d.zhukov/CLionProjects/HopfieldNN/slim_digit/learn_0.png",
                                      &width, &height, &channels, 0);
    unsigned char *learn1 = stbi_load("/Users/il.d.zhukov/CLionProjects/HopfieldNN/slim_digit/learn_1.png",
                                      &width, &height, &channels, 0);
    unsigned char *learn2 = stbi_load("/Users/il.d.zhukov/CLionProjects/HopfieldNN/slim_digit/learn_2.png",
                                      &width, &height, &channels, 0);
    unsigned char *learn3 = stbi_load("/Users/il.d.zhukov/CLionProjects/HopfieldNN/slim_digit/learn_3.png",
                                      &width, &height, &channels, 0);
    unsigned char *learn4 = stbi_load("/Users/il.d.zhukov/CLionProjects/HopfieldNN/slim_digit/learn_4.png",
                                      &width, &height, &channels, 0);
    unsigned char *learn5 = stbi_load("/Users/il.d.zhukov/CLionProjects/HopfieldNN/slim_digit/learn_5.png",
                                      &width, &height, &channels, 0);
    unsigned char *learn6 = stbi_load("/Users/il.d.zhukov/CLionProjects/HopfieldNN/slim_digit/learn_6.png",
                                      &width, &height, &channels, 0);
    unsigned char *learn7 = stbi_load("/Users/il.d.zhukov/CLionProjects/HopfieldNN/slim_digit/learn_7.png",
                                      &width, &height, &channels, 0);
    unsigned char *learn8 = stbi_load("/Users/il.d.zhukov/CLionProjects/HopfieldNN/slim_digit/learn_8.png",
                                      &width, &height, &channels, 0);
    unsigned char *learn9 = stbi_load("/Users/il.d.zhukov/CLionProjects/HopfieldNN/slim_digit/learn_9.png",
                                      &width, &height, &channels, 0);

    unsigned char *testBy = stbi_load("/Users/il.d.zhukov/CLionProjects/HopfieldNN/thick_digit/learn_7.png",
                                      &width, &height, &channels, 0);


    std::vector<unsigned char *> learnLinks{learn0, learn1, learn2, learn3, learn4, learn5, learn6, learn7, learn8, learn9};
    std::vector<unsigned char *> testLink{testBy};

    std::vector<std::vector<int>> learnImages = readImageForLearn(learnLinks, width, height);
    std::vector<std::vector<double>> weight = trainWithNorm(learnImages);

    std::vector<int> testByImage = readImageForTest(testLink, width, height)[0];

    std::cout << "learned" << std::endl;
    recognize(testByImage, weight, width, height);


    return 0;
}