#include <iostream>
#include "draw.h"
#include "utils.h"


const int TILE_WIDTH = 2;

__constant__ float SOFTENING_RATE;
__constant__ float DELTA_TIME;
extern __shared__ float3 sharedPositions[];

__device__ float2 CalculateAccelerationBetweenTwoParticles(float2 acceleration, float3 firstPosition, float3 secondPosition) {
    float2 delta;
    delta.x = firstPosition.x - secondPosition.x;
    delta.y = firstPosition.y - secondPosition.y;

    float distance = sqrtf((delta.x * delta.x + delta.y * delta.y) + SOFTENING_RATE);
    float force = secondPosition.z / (distance * distance * distance);

    acceleration.x += delta.x * force;
    acceleration.y += delta.y * force;

    return acceleration;
}

__device__ float2 CalculateAccelerationInTile(float3 position, float2 acceleration) {
    for (int i = 0; i < blockDim.x * TILE_WIDTH / 4; i++) {
        acceleration = CalculateAccelerationBetweenTwoParticles(acceleration, sharedPositions[4*i], position);
        acceleration = CalculateAccelerationBetweenTwoParticles(acceleration, sharedPositions[4*i+1], position);
        acceleration = CalculateAccelerationBetweenTwoParticles(acceleration, sharedPositions[4*i+2], position);
        acceleration = CalculateAccelerationBetweenTwoParticles(acceleration, sharedPositions[4*i+3], position);
    }

//    for (int i = 0; i < blockDim.x * TILE_WIDTH; i += 4) {
//        acceleration = CalculateAccelerationBetweenTwoParticles(acceleration, sharedPositions[i], position);
//        acceleration = CalculateAccelerationBetweenTwoParticles(acceleration, sharedPositions[i+1], position);
//        acceleration = CalculateAccelerationBetweenTwoParticles(acceleration, sharedPositions[i+2], position);
//        acceleration = CalculateAccelerationBetweenTwoParticles(acceleration, sharedPositions[i+3], position);
//    }

    return acceleration;
}

__device__ float2 CalculateAcceleration(float3 position, float3 *positions, int particlesCount) {
    float2 acceleration = {0.0f, 0.0f};

    for (int i = 0; i < particlesCount / blockDim.x / TILE_WIDTH; i++) {
        for (int j = 0; j < TILE_WIDTH; j++) {
            sharedPositions[blockDim.x * j + threadIdx.x] = positions[(TILE_WIDTH * i + j) * blockDim.x + threadIdx.x];
        }
        __syncthreads();
        acceleration = CalculateAccelerationInTile(position, acceleration);
        __syncthreads();
    }

    return acceleration;
}

__global__ void Update(float3 *newPositions, float2 *newVelocities, float3 *oldPositions, float2 *oldVelocities, int particlesCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float3 position = oldPositions[index];
    float2 velocity = oldVelocities[index];

    float2 acceleration = CalculateAcceleration(position, oldPositions, particlesCount);

    velocity.x += acceleration.x * DELTA_TIME;
    velocity.y += acceleration.y * DELTA_TIME;

    position.x += velocity.x * DELTA_TIME;
    position.y += velocity.y * DELTA_TIME;

    newPositions[index] = position;
    newVelocities[index] = velocity;
}

void Run(int particlesCount, int blockSize, float3 *hostPositions, float deltaTime, float softeningRate, int iterationsCount, bool validate = false) {
    int gridSize = particlesCount / blockSize;
    int sharedMemSize = blockSize * TILE_WIDTH * sizeof(float3);

    float3 *deviceNewPositions;
    float3 *deviceOldPositions;
    float2 *deviceNewVelocities;
    float2 *deviceOldVelocities;

    CUDA_CHECK(cudaMalloc((float3 **) &deviceNewPositions, particlesCount * sizeof(float3)));
    CUDA_CHECK(cudaMalloc((float3 **) &deviceOldPositions, particlesCount * sizeof(float3)));
    CUDA_CHECK(cudaMalloc((float2 **) &deviceNewVelocities, particlesCount * sizeof(float2)));
    CUDA_CHECK(cudaMalloc((float2 **) &deviceOldVelocities, particlesCount * sizeof(float2)));

    CUDA_CHECK(cudaMemcpyToSymbol(DELTA_TIME, &deltaTime, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(SOFTENING_RATE, &softeningRate, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(deviceOldPositions, hostPositions, particlesCount * sizeof(float3), cudaMemcpyHostToDevice));

    #ifdef DRAW
    sf::RenderWindow window(sf::VideoMode(800, 800), "N-Body simulation");
    #endif

    for (int i = 0; i < iterationsCount; i++) {
        Update<<<gridSize, blockSize, sharedMemSize>>>(deviceNewPositions, deviceNewVelocities, deviceOldPositions, deviceOldVelocities, particlesCount);

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(deviceNewPositions, deviceOldPositions);
        std::swap(deviceNewVelocities, deviceOldVelocities);

        #ifdef DRAW
        if (i % 10 == 0) {
            cudaMemcpy(hostPositions, deviceNewPositions, particlesCount * sizeof(float3), cudaMemcpyDeviceToHost);
            Draw(&window, hostPositions, particlesCount);
            if (validate) {
                std::cout << hostPositions[0].x << std::endl;
            }
        }
        #endif
    }

    CUDA_CHECK(cudaFree(deviceNewPositions));
    CUDA_CHECK(cudaFree(deviceOldPositions));
    CUDA_CHECK(cudaFree(deviceNewVelocities));
    CUDA_CHECK(cudaFree(deviceOldVelocities));
}