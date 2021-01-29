#include <iostream>
#include <chrono>

void Run(int particlesCount, int blockSize, float3 *hostPositions, float deltaTime, float softeningRate, int iterationsCount, bool validate = false);

float3 *GenerateParticles(int particlesCount) {
    auto *positions = new float3[particlesCount];

    for (int i = 0; i < particlesCount; i++) {
        positions[i].x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 0.5f + 0.25f;
        positions[i].y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 0.5f + 0.25f;
        positions[i].z = 1.0f;
    }

    return positions;
}

void Benchmark(int particlesCount, int blockSize, int iterationsCount) {
    float deltaTime = 0.000001;
    float softeningRate = 0.00001;
    float3 *positions = GenerateParticles(particlesCount);

    auto start = std::chrono::high_resolution_clock::now();

    Run(particlesCount, blockSize, positions, deltaTime, softeningRate, iterationsCount);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Time: " << elapsed.count() << std::endl;

    delete[] positions;
}

int main(int argc, char **argv) {
    srand(static_cast <unsigned> (time(nullptr)));
    int particlesCount = 8192;
    int blockSize = 1024;
    int iterationsCount = 200000;

    if (argc > 1) {
        particlesCount = std::stoi(argv[1]);
    }
    if (argc > 2) {
        blockSize = std::stoi(argv[2]);
    }
    if (argc > 3) {
        iterationsCount = std::stoi(argv[3]);
    }

    if (particlesCount % blockSize != 0) {
        std::cout << "Number of particles must be divisible by block size" << std::endl;
        return -1;
    }
    if (blockSize % 4 != 0) {
        std::cout << "Block size must be divisible by 4" << std::endl;
        return -1;
    }

    Benchmark(particlesCount, blockSize, iterationsCount);

    return 0;
}

