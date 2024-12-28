// headers 
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>
#include <chrono>
#include <random>
#include <algorithm> // For std::max

#ifdef _OPENMP
#include <omp.h>    // For OpenMP parallel processing
#endif

#if defined __linux__ || defined __APPLE__
// "Compiled for linux"
#else
// Windows doesn't define these values by default but linux does
#define M_PI 3.141592653589793
#define INFINITY 1e8
#endif

#define MAX_RAY_DEPTH 10
#define EPSILON 1e-4 // Precision control

// A- template class for 3-dimensional vector 
template<typename T>
class Vec3 
{
    public: 
        T x, y, z; // Components of vector

        // Default constructor initializes components to zero
        Vec3() : x(T(0)), y(T(0)), z(T(0)) {}

        // Constructor initializing all values to scalars
        Vec3(T xx) : x(xx), y(xx), z(xx) {}

        // Constructor with individual component values
        Vec3(T xx,T yy,T zz) : x(xx), y(yy), z(zz) {}

        // Normalizes the vector to unit length. Checks for zero length to prevent division by zero.
        Vec3& normalize() {
            T nor2 = length2();
            if (nor2 > 0) {
                T invNor = 1 / sqrt(nor2);
                x *= invNor, y *= invNor, z *= invNor;
            }
            return *this;
        }

        // Operator Overloads:

        // Multiplication by Scalar and return a scaled vector
        Vec3<T> operator * (const T &f) const { return Vec3<T>(x * f, y * f, z * f); }

        // component wise multiplication and returns the product of two vectors
        Vec3<T> operator * (const Vec3<T> &v) const { return Vec3<T>(x * v.x, y * v.y, z * v.z); }

        // Dot Product, returns the scalar dot product of two vectors
        T dot(const Vec3<T> &v) const { return x * v.x + y * v.y + z * v.z; }

        // SUbtraction, returns the difference between two vectors
        Vec3<T> operator - (const Vec3<T> &v) const { return Vec3<T>(x - v.x, y - v.y, z - v.z); }

        // Addition, returns the sum of two vectors
        Vec3<T> operator + (const Vec3<T> &v) const { return Vec3<T>(x + v.x, y + v.y, z + v.z); }

        // Addition assignment, adds a vector to this vector.
        Vec3<T>& operator += (const Vec3<T> &v) { x += v.x, y += v.y, z += v.z; return *this; }

        // Multiplication assignment, multiplies this vector by another vector component-wise.
        Vec3<T>& operator *= (const Vec3<T> &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }

        // Negation, returns the negation of the vector.
        Vec3<T> operator - () const { return Vec3<T>(-x, -y, -z); }

        // Returns the squared length of the vector, useful for distance comparisons.
        T length2() const { return x * x + y * y + z * z; }

        // Returns the length of the vector.
        T length() const { return sqrt(length2()); }

        //overload os stream operator for easy printing
        friend std::ostream & operator << (std::ostream &os, const Vec3<T> &v) {
            os << "[" << v.x << " " << v.y << " " << v.z << "]";
            return os;
        }
};


// Defining Vec3f as Vec3 of float components
typedef Vec3<float> Vec3f;

class Sphere {
public:
    Vec3f center;             // Position of the sphere
    float radius, radius2;    // Sphere radius and squared radius for optimization
    Vec3f surfaceColor;       // Color of the sphere surface
    Vec3f emissionColor;      // Emission color, for light emitting objects
    float transparency;       // Transparency level [0,1]
    float reflection;         // Reflection level [0,1]

    // Constructor with error checking for radius and defaults for optional parameters.
    Sphere(
        const Vec3f &c,
        const float &r,
        const Vec3f &sc,
        const float &refl = 0,
        const float &transp = 0,
        const Vec3f &ec = Vec3f(0)) :
        center(c), radius(r), surfaceColor(sc), emissionColor(ec),
        transparency(transp), reflection(refl) {

        // Ensure radius is positive; if not, set to a default small positive value.
        if (radius <= 0.0) {
            std::cerr << "Invalid radius value. Setting to 0.1." << std::endl;
            radius = 0.1;
        }
        radius2 = radius * radius; // Precompute squared radius
    }

    /**
     * Computes a ray-sphere intersection using the geometric solution.
     * 
     * @param rayorig Origin of the ray.
     * @param raydir Direction of the ray, assumed to be normalized.
     * @param t0 The nearer intersection distance from the ray origin (if intersection occurs).
     * @param t1 The farther intersection distance from the ray origin (if intersection occurs).
     * @return True if there is an intersection, false otherwise.
     * 
     * This method checks if a ray intersects with the sphere. The geometric solution involves finding
     * the point where the ray intersects the sphere's surface, which can occur at two points, entering
     * and exiting. If no intersection is found, the function returns false.
    */
    bool intersect(const Vec3f &rayorig, const Vec3f &raydir, float &t0, float &t1) const {
        Vec3f l = center - rayorig; // Vector from ray origin to sphere center
        float tca = l.dot(raydir); // Closest approach along the ray direction
        if (tca < 0) return false; // Sphere is behind the ray

        float d2 = l.dot(l) - tca * tca; // Squared distance from sphere center to the ray
        if (d2 > radius2) return false; // Ray misses the sphere
        
        float thc = sqrt(radius2 - d2); // Half-chord distance
        t0 = tca - thc; // Distance to the first intersection point
        t1 = tca + thc; // Distance to the second intersection point

        // Ensure t0 is less than t1
        if (t0 > t1) std::swap(t0, t1);

        return true; // Intersection occurs
    }
};

/**
 * Computes the weighted average of two values.
 * 
 * @param a The first value.
 * @param b The second value.
 * @param mix The weight applied to b, clamped between 0 and 1.
 * @return The weighted average of a and b, biased according to the mix parameter.
 * 
 * This function ensures that the mix parameter is clamped to the range [0,1],
 * preventing extrapolation beyond the intended range and ensuring the function
 * always returns a weighted average within the bounds of a and b.
*/
float mix(const float &a, const float &b, const float &mix) {
    float clampedMix = std::max(0.0f, std::min(1.0f, mix)); // Clamp mix to [0, 1]
    return b * clampedMix + a * (1 - clampedMix);
}

/**
 * Raises each component of the vector to the power provided.
 * 
 * @param v The input vector.
 * @param exponent The exponent to which each component of the vector is raised.
 * @return A vector where each component is the result of raising the original
 *         component to the specified exponent.
 */
Vec3f pow(const Vec3f &v, float exponent) {
    return Vec3f(std::pow(v.x, exponent), std::pow(v.y, exponent), std::pow(v.z, exponent));
}


/**
 * The main trace function for ray tracing. It takes a ray defined by its origin
 * and direction and tests if this ray intersects with any geometry in the scene.
 * For intersections, it computes the intersection point, the normal at the
 * intersection, and shades this point based on surface properties and the
 * scene's lighting.
 *
 * Shading depends on the surface's properties, such as whether it's transparent,
 * reflective, or diffuse. The function returns the color for the ray, which is
 * the color of the object at the intersection point if there's an intersection,
 * or the background color if there isn't.
 *
 * @param rayorig The origin of the ray.
 * @param raydir The direction of the ray, assumed to be normalized.
 * @param spheres A vector of spheres in the scene to test for intersections.
 * @param depth The current recursion depth, used to limit reflections and refractions.
 * @return The color computed for the ray.
 *
 * trace function include precision control through an epsilon value to manage
 * floating-point precision issues, optimized shadow checks to reduce unnecessary
 * computations, and handling edge cases in refraction calculations.
*/

Vec3f trace(
    const Vec3f &rayorig,
    const Vec3f &raydir,
    const std::vector<Sphere> &spheres,
    const int &depth)
{
    float tnear = INFINITY;
    const Sphere* sphere = nullptr;
    // Find the intersection of this ray with the spheres in the scene
    for (size_t i = 0; i < spheres.size(); ++i) {
        float t0 = INFINITY, t1 = INFINITY;
        if (spheres[i].intersect(rayorig, raydir, t0, t1)) {
            if (t0 < EPSILON) t0 = t1;
            if (t0 < tnear) {
                tnear = t0;
                sphere = &spheres[i];
            }
        }
    }
    // If there's no intersection, return black or background color
    if (!sphere) {
        // std::cerr << "Ray missed all objects, returning background color.\n";
        return Vec3f(1.0, 1.0, 1.0); // Ensure this is Vec3f(1, 1, 1) for white
    } 
    Vec3f surfaceColor = 0; // Color of the ray/surface of the object intersected by the ray
    Vec3f phit = rayorig + raydir * tnear; // Point of intersection
    Vec3f nhit = phit - sphere->center; // Normal at the intersection point
    nhit.normalize(); // Normalize normal direction
    // std::cerr << "Hit object, color: " << surfaceColor << ", position: " << phit << "\n";
    // Inside the sphere check
    bool inside = false;
    if (raydir.dot(nhit) > 0) nhit = -nhit, inside = true;

    // Handling reflections and refractions for transparent and reflective surfaces
    if ((sphere->transparency > 0 || sphere->reflection > 0) && depth < MAX_RAY_DEPTH) {
        float facingratio = -raydir.dot(nhit);
        float fresneleffect = mix(std::pow(1 - facingratio, 3), 1, 0.1); // Tweak the mix value to adjust the effect
        Vec3f refldir = raydir - nhit * 2 * raydir.dot(nhit);
        refldir.normalize();
        Vec3f reflection = trace(phit + nhit * EPSILON, refldir, spheres, depth + 1);
        Vec3f refraction = 0;
        // Refraction
        if (sphere->transparency) {
            float ior = 1.1, eta = inside ? ior : 1 / ior;
            float cosi = -nhit.dot(raydir);
            float k = 1 - eta * eta * (1 - cosi * cosi);
            if (k >= 0) { // Total internal reflection check
                Vec3f refrdir = raydir * eta + nhit * (eta * cosi - std::sqrt(k));
                refrdir.normalize();
                refraction = trace(phit - nhit * EPSILON, refrdir, spheres, depth + 1);
            }
        }
        surfaceColor = (reflection * fresneleffect + refraction * (1 - fresneleffect) * sphere->transparency) * sphere->surfaceColor;
    } else {
        // It's a diffuse object, no need to raytrace any further for reflections
        for (size_t i = 0; i < spheres.size(); ++i) {
            if (spheres[i].emissionColor.x > 0) {
                // This is a light source
                Vec3f transmission = 1;
                Vec3f lightDirection = spheres[i].center - phit;
                lightDirection.normalize();
                for (size_t j = 0; j < spheres.size(); ++j) {
                    float t0 = INFINITY, t1 = INFINITY; // Declare t0 and t1 for shadow checks
                    if (i != j && spheres[j].intersect(phit + nhit * EPSILON, lightDirection, t0, t1)) {
                        transmission = 0;
                        break; // Optimized shadow check - break early if shadow is confirmed
                    }
                }
                surfaceColor += sphere->surfaceColor * transmission * std::max(0.f, nhit.dot(lightDirection)) * spheres[i].emissionColor;
            }
        }
    }
    
    return surfaceColor + sphere->emissionColor;
}


/**
 * Renders a scene composed of spheres to a PPM image file.
 * 
 * @param spheres The list of spheres to render.
 * @param width The width of the output image in pixels.
 * @param height The height of the output image in pixels.
 * @param fov The field of view in degrees for the camera.
 */
void render(const std::vector<Sphere>& spheres, unsigned width, unsigned height, float fov) {
    std::vector<Vec3f> image(width * height);
    float aspectRatio = static_cast<float>(width) / height;
    float angle = std::tan(M_PI * 0.5 * fov / 180.);

    // Parallel processing using OpenMP (optional)
    #pragma omp parallel for schedule(dynamic)
    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            float xx = (2 * ((x + 0.5) * (1 / float(width))) - 1) * angle * aspectRatio;
            float yy = (1 - 2 * ((y + 0.5) * (1 / float(height)))) * angle;
            Vec3f rayDir(xx, yy, -1);
            rayDir.normalize();
            size_t idx = y * width + x;
            image[idx] = trace(Vec3f(0), rayDir, spheres, 0);
        }
    }

    // Save the rendered image to a PPM file
    std::ofstream ofs("./untitled.ppm", std::ios::out | std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: Could not open output file." << std::endl;
        return;
    }
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (const auto& pix : image) {
        Vec3f correctedColor = pow(pix, 1.0f / 2.2f); // Apply gamma correction
        ofs << static_cast<unsigned char>(std::min(1.f, correctedColor.x) * 255)
            << static_cast<unsigned char>(std::min(1.f, correctedColor.y) * 255)
            << static_cast<unsigned char>(std::min(1.f, correctedColor.z) * 255);
        }
}


/**
 * The entry point of the ray tracing application.
 * 
 * This program initializes a scene with predefined spheres and a light source,
 * then renders the scene to an image file. The rendering process's execution time
 * is measured and displayed.
 * 
 * The scene setup includes five spheres with various properties (position, radius,
 * surface color, reflectivity, transparency) and a light source modeled as a sphere
 * with emission color. After setting up the scene, the render function is called to
 * generate the image, and the execution time for rendering the scene is measured
 * and output to the console.
 * 
 * Usage:
 *   <executable_name>
 * 
 * Note: This program does not currently take command-line arguments for scene configuration.
 * Future versions could include this feature for enhanced flexibility.
 * 
 * @param argc Number of command-line arguments (unused in this version).
 * @param argv Array of command-line argument strings (unused in this version).
 * @return Returns 0 upon successful execution.
*/
int main(int argc, char **argv) {
    // Seed for random number generation, used in various parts of the rendering
    #if defined(__unix__) || defined(__APPLE__)
    srand48(13);
    #elif defined(_WIN32)
    std::mt19937_64 rng(std::random_device{}());
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 eng(rd()); // Seed the generator
    std::uniform_real_distribution<> distr(0.0, 1.0); // Define the range
    #endif

    // Record the start time of the rendering process
    auto startTime = std::chrono::high_resolution_clock::now();

    // Initialize the scene with spheres and a light source
    std::vector<Sphere> spheres;
    // // Adding spheres to the scene: position, radius, surface color, reflectivity, transparency, emission color
    // spheres.emplace_back(Sphere(Vec3f(0.0, -10004, -20), 10000, Vec3f(0.20, 0.20, 0.20), 0, 0.0));
    // spheres.emplace_back(Sphere(Vec3f(0.0, 0, -20), 4, Vec3f(1.00, 0.32, 0.36), 1, 0.5));
    // spheres.emplace_back(Sphere(Vec3f(5.0, -1, -15), 2, Vec3f(0.90, 0.76, 0.46), 1, 0.0));
    // spheres.emplace_back(Sphere(Vec3f(5.0, 0, -25), 3, Vec3f(0.65, 0.77, 0.97), 1, 0.0));
    // spheres.emplace_back(Sphere(Vec3f(-5.5, 0, -15), 3, Vec3f(0.90, 0.90, 0.90), 1, 0.0));
    // // Adding a light source modeled as a sphere
    // spheres.emplace_back(Sphere(Vec3f(0.0, 20, -30), 3, Vec3f(0.00, 0.00, 0.00), 0, 0.0, Vec3f(3)));

    // Large ground sphere for more hit chances
    spheres.emplace_back(Sphere(Vec3f(0.0, -10004, -20), 10000, Vec3f(0.20, 0.20, 0.20), 0, 0.0));

    // A grid of smaller spheres
    int sphereGridSize = 5; // 5x5 grid
    for (int i = -sphereGridSize; i <= sphereGridSize; ++i) {
        for (int j = -sphereGridSize; j <= sphereGridSize; ++j) {
            Vec3f position = Vec3f(i * 3.0f, 0, j * 3.0f - 20);
            Vec3f color = Vec3f(distr(eng) + 0.5, distr(eng) + 0.5, distr(eng) + 0.5);
            spheres.emplace_back(Sphere(position, 2, color, distr(eng), distr(eng)));
        }
    }

    // Light sources
    spheres.emplace_back(Sphere(Vec3f(0.0, 20, -30), 3, Vec3f(0.00, 0.00, 0.00), 0, 0.0, Vec3f(3)));
    spheres.emplace_back(Sphere(Vec3f(-3.0, 5, -10), 3, Vec3f(0.00, 0.00, 0.00), 0, 0.0, Vec3f(3))); // Additional light

    // Render the scene to an image
    unsigned width = 1200, height = 800; // which later over ridden by command line arguments
    float fov = 30;
    render(spheres, width, height, fov);

    // Record the end time of the rendering process and calculate the elapsed time
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - startTime;

    // Output the execution time to the console
    std::cout << "Rendering finished in " << elapsedSeconds.count() << " seconds.\n";

    return 0;

}