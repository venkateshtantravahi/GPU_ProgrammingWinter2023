//
// Created by vtant on 4/7/2024.
// Defines the `ray` class, representing a ray in 3D space.
// A ray is defined by its origin point and direction vector.
//

#ifndef RAYTRACING_RAY_H
#define RAYTRACING_RAY_H

#include "vec3.h" // Include the vec3 class for 3D vector operations

// The `ray` class encapsulates a ray in 3D space.
class ray {
public:
    // Default constructor. Creates an uninitialized ray.
    __device__ ray() {}

    // Constructor that initializes a ray with an origin (`a`) and direction (`b`).
    // @param a origin point of the ray.
    // @param b The direction vector of the ray, not necessarily normalized.
    __device__ ray(const vec3& a, const vec3& b) {A = a; B = b; }

    // Returns the origin point of the ray.
    // @return The origin point as a `vec3`.
    __device__ vec3 origin() const {return A; }

    // Returns the direction vector of the ray.
    // @return The direction vector as a `vec3`.
    __device__ vec3 direction() const { return B; }

    // Computes a point along the ray at a given parameter `t`.
    // @param t The parameter at which to compute the point along the ray.
    // @return The point along the ray as a `vec3`.
    __device__ vec3 point_at_parameter(float t) const { return A + t*B; }

    vec3 A; // The origin point of the ray
    vec3 B; // The direction vector of the ray
};

#endif //RAYTRACING_RAY_H
