//
// Created by vtant on 4/7/2024.
//

#ifndef RAYTRACING_VEC3_H
#define RAYTRACING_VEC3_H


#include <math.h>
#include <stdlib.h>
#include <iostream>

// A 3-dimensional vector class for representing vectors and colors
class vec3 {

public:
    // Default constructor
    __host__ __device__ vec3() {}
    // Constructor with initial values
    __host__ __device__ vec3(float a0, float a1, float a2) {a[0] = a0; a[1] = a1; a[2] = a2; }
    // Accessors for vector components
    __host__ __device__ inline float x() const {return a[0]; }
    __host__ __device__ inline float y() const {return a[1]; }
    __host__ __device__ inline float z() const {return a[2]; }
    // Color accessors, treating the vector as an RGB color
    __host__ __device__ inline float r() const {return a[0]; }
    __host__ __device__ inline float g() const {return a[1]; }
    __host__ __device__ inline float b() const {return a[2]; }

    // Operator overloads for vector arithmetic
    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-a[0], -a[1], -a[2]); }
    __host__ __device__ inline float operator[](int i) const {return a[i]; }
    __host__ __device__ inline float& operator[](int i) { return a[i]; };

    // Compound assignment operators for vector arithmetic
    __host__ __device__ inline vec3& operator+=(const vec3 &v2);
    __host__ __device__ inline vec3& operator-=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const vec3 &v2);
    __host__ __device__ inline vec3& operator/=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const float t);
    __host__ __device__ inline vec3& operator/=(const float t);

    // Vector magnitude calculations
    __host__ __device__ inline float length() const { return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]); }
    __host__ __device__ inline float squared_length() const { return a[0]*a[0] + a[1]*a[1] + a[2]*a[2]; }
    // Normalizes this vector
    __host__ __device__ inline void make_unit_vector();

    // Array of float to store the vector components
    float a[3];
};

// Stream insertion operators for input and output
inline std::istream& operator>>(std::istream &is, vec3 &t) {
    is >> t.a[0] >> t.a[1] >> t.a[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
    os << t.a[0] << " " << t.a[1] << " " << t.a[2];
    return os;
}

// Makes this vector have a magnitude of 1
__host__ __device__ inline void vec3::make_unit_vector() {
    float k = 1.0 / sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    a[0] *= k; a[1] *= k; a[2] *= k;
}

// Overloads for vector arithmetic operations
__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.a[0] + v2.a[0], v1.a[1] + v2.a[1], v1.a[2] + v2.a[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.a[0] - v2.a[0], v1.a[1] - v2.a[1], v1.a[2] - v2.a[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.a[0] * v2.a[0], v1.a[1] * v2.a[1], v1.a[2] * v2.a[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.a[0] / v2.a[0], v1.a[1] / v2.a[1], v1.a[2] / v2.a[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t*v.a[0], t*v.a[1], t*v.a[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return vec3(v.a[0]/t, v.a[1]/t, v.a[2]/t);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
    return vec3(t*v.a[0], t*v.a[1], t*v.a[2]);
}

// Dot product of two vectors
__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2) {
    return v1.a[0] *v2.a[0] + v1.a[1] *v2.a[1]  + v1.a[2] *v2.a[2];
}

// Cross product of two vectors
__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    return vec3( (v1.a[1]*v2.a[2] - v1.a[2]*v2.a[1]),
                 (-(v1.a[0]*v2.a[2] - v1.a[2]*v2.a[0])),
                 (v1.a[0]*v2.a[1] - v1.a[1]*v2.a[0]));
}


__host__ __device__ inline vec3& vec3::operator+=(const vec3 &v){
    a[0]  += v.a[0];
    a[1]  += v.a[1];
    a[2]  += v.a[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3 &v){
    a[0]  *= v.a[0];
    a[1]  *= v.a[1];
    a[2]  *= v.a[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3 &v){
    a[0]  /= v.a[0];
    a[1]  /= v.a[1];
    a[2]  /= v.a[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v) {
    a[0]  -= v.a[0];
    a[1]  -= v.a[1];
    a[2]  -= v.a[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t) {
    a[0]  *= t;
    a[1]  *= t;
    a[2]  *= t;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t) {
    float k = 1.0/t;

    a[0]  *= k;
    a[1]  *= k;
    a[2]  *= k;
    return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

#endif //RAYTRACING_VEC3_H