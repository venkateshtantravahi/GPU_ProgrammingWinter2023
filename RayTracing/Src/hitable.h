//
// Created by vtant on 4/7/2024.
// Defines the abstract base class `hitable` and the structure `hit_record`.
// `hitable` is an interface for objects that can be intersected by a ray.
// `hit_record` contains information about a ray-object intersection.
//

#ifndef RAYTRACING_HITABLE_H
#define RAYTRACING_HITABLE_H

#include "ray.h" // Include the ray class for ray operations

// Forward declaration of the `material` class to be used in the `hit_record`.
class material;

// The `hit_record` struct stores information about the intersection of a ray with an object.
struct hit_record {
    float t; // The parameter `t` at the hit point along the ray
    vec3 p; // The point at which the ray hits the object
    vec3 normal; // The normal vector at the hit point
    material *mat_ptr; // A pointer to the material of the object hit
};

// The `hitable` class is an abstract base class for objects that can be intersected by a ray.
class hitable {
public:
    // Pure virtual function to determine if a ray hits an object within a range specified by `t_min` and `t_max`.
    // @param r The ray being cast through the scene.
    // @param t_min The minimum value of `t` to consider a hit.
    // @param t_max The maximum value of `t` to consider a hit.
    // @param rec A reference to a `hit_record` structure where hit information will be stored.
    // @return `true` if the ray hits the object between `t_min` and `t_max`, `false` otherwise.
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif //RAYTRACING_HITABLE_H
