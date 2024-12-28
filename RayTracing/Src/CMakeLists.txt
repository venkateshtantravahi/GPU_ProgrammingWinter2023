cmake_minimum_required(VERSION 3.27)
project(RayTracing CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(RayTracing main.cu
        vec3.h
        hitable.h
        ray.h
        hitable_list.h
        sphere.h
        material.h
        camera.h)

set_target_properties(RayTracing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
