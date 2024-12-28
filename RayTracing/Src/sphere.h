//
// Created by vtant on 4/7/2024.
// Defines the `sphere` class, a concrete implementation of `hitable`.
// Represents a sphere in 3D space that can be intersected by a ray.
//

#ifndef RAYTRACING_SPHERE_H
#define RAYTRACING_SPHERE_H

#include "hitable.h"

// The `sphere` class represents a sphere as a hitable object in the scene.
class sphere: public hitable {
public:
    // Default constructor
    __device__ sphere() {}

    // Constructor that initializes a sphere with its center, radius, and material.
    // @param cen The center point of the sphere.
    // @param r The radius of the sphere.
    // @param m A pointer to the material of the sphere.
    __device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m) {};

    // Determines if this sphere is hit by the given ray within the specified range.
    // @param r The ray being cast through the scene.
    // @param tmin The minimum value of `t` to consider a hit.
    // @param tmax The maximum value of `t` to consider a hit.
    // @param rec A reference to a `hit_record` structure where hit information will be stored if a hit occurs.
    // @return `true` if the ray hits the sphere between `tmin` and `tmax`, `false` otherwise.
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    vec3 center; // The center of the sphere
    float radius; // The radius of the sphere
    material *mat_ptr; // A pointer to the material of the sphere
};

// Implementation of the `hit` method for the `sphere` class.
// Determines if a ray intersects with this sphere within specified limits of t.
__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    // Vector from ray origin to the center of the sphere.
    vec3 oc = r.origin() - center;

    // Coefficients for the quadratic equation (A*t^2 + B*t + C = 0) derived from the ray-sphere intersection formula.
    float a = dot(r.direction(), r.direction()); // Coefficient A: dot product of ray direction with itself (|d|^2)
    float b = dot(oc, r.direction()); // Coefficient B: 2*dot product of OC vector with ray direction, factor of 2 omitted.
    float c = dot(oc, oc) - radius*radius; // Coefficient C: dot product of OC vector with itself minus square of sphere's radius.

    // Discriminant of the quadratic equation (B^2 - 4AC), part of the solution to the quadratic equation.
    // Determines the nature of the roots of the equation (how many intersection points).
    float discriminant = b*b - a*c;

    // If the discriminant is positive, there are two distinct intersection points (ray intersects the sphere).
    if (discriminant > 0) {

        // Check the first root of the quadratic equation.
        // Temporarily store the intersection point closer to the ray's origin.
        float temp = (-b - sqrt(discriminant))/a;

        // If this point lies within the acceptable range of t (t_min to t_max),
        // it is considered a valid hit, and the hit_record is updated accordingly.
        if (temp < t_max && temp > t_min) {
            rec.t = temp;   // Store the parameter t at which the ray intersects the sphere.
            rec.p = r.point_at_parameter(rec.t); // Calculate and store the point of intersection.
            rec.normal = (rec.p - center) / radius; // Compute and store the normal at the point of intersection.
            rec.mat_ptr = mat_ptr;  // Associate the material of the sphere with the hit.
            return true;    // Return true to indicate a hit.
        }

        // Check the second root of the quadratic equation (possible second intersection point).
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;   // Update hit_record with the parameter t for the second intersection.
            rec.p = r.point_at_parameter(rec.t);    // Calculate and store the point of intersection for the second root.
            rec.normal = (rec.p - center) / radius ;    // Compute and store the normal at the point of intersection.
            rec.mat_ptr = mat_ptr;  // Associate the material of the sphere with the hit.
            return true;    // Return true as the ray hits the sphere.
        }
    }
    return false;   // If the discriminant is non-positive or no valid t is found within the range, return false (no hit).
}

#endif //RAYTRACING_SPHERE_H
