//
// Created by vtant on 4/7/2024.
// Defines the `hitable_list` class, a concrete implementation of `hitable` that represents a list of hitable objects.
// This class allows for multiple objects to be tested for intersection with a ray.
//

#ifndef RAYTRACING_HITABLE_LIST_H
#define RAYTRACING_HITABLE_LIST_H

#include "hitable.h"

// `hitable_list` is a collection of objects that implement the `hitable` interface.
class hitable_list: public hitable {
public:
    // Default constructor
    __device__ hitable_list() {}

    // Constructor that initializes the list with a given array of hitable objects and the number of objects.
    // @param l Pointer to the array of hitable object pointers.
    // @param n The number of hitable objects in the array.
    __device__ hitable_list(hitable **l, int n) {list=l; list_size=n;}

    // Checks if a ray hits any object in the list within the specified range.
    // @param r The ray being cast through the scene.
    // @param tmin The minimum value of `t` to consider a hit.
    // @param tmax The maximum value of `t` to consider a hit.
    // @param rec A reference to a `hit_record` structure where hit information will be stored if a hit occurs.
    // @return `true` if the ray hits any object in the list between `tmin` and `tmax`, `false` otherwise.
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

    hitable **list; // Pointer to an array of pointers to hitable objects
    int list_size; // The number of hitable objects in the list
};

// Implementation of the `hit` method for the `hitable_list` class.
__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    // Loop over all objects in the list, checking for intersections.
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t; // Update the closest hit found so far
            rec = temp_rec; // Update the hit record with the closest hit
        }
    }
    return hit_anything;
}

#endif //RAYTRACING_HITABLE_LIST_H
