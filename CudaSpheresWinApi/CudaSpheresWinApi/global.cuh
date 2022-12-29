#pragma once
#include "utilities.cuh"
#include <memory>


const float RATIO = 16.0f / 9.0f;
const int HEIGHT = 720;
const int WIDTH = RATIO * HEIGHT;

const bool GPU_RENDER = true;

const int framesX = 16;
const int framesY = 16;

const int SPHERE_COUNT = 2000;
const int LIGHT_COUNT = 20;
const float SPREAD = 5.5f;
const int SAMPLES_PER_PIXEL = 5;
const float MOUSE_SPEED = 8.0f;
const float KEY_BATCH = 0.25f;

__device__ const float CAMERA_VIEWPORT = 2.0f;


struct ray
{
	float3 origin;
	float3 direction;

	ray() {}

	__host__ __device__ ray(const float3& _origin, const float3& _dir) :
		origin(_origin), direction(_dir) {}

	__host__ __device__ float3 at(float t) const {
		return origin + t * direction;
	}
};

struct hit_record {
	float3 p; // punkt przeciecia raya ze sfera
	float3 normal; // wektor normalny
	float3 v; // wektor do kamery
	float t; // parametr t raya dla ktorego sie przecina z kula
	int index; // indeks kuli ktorej dotyczy
};

struct colors
{
	float* r;
	float* g;
	float* b;
};

struct positions
{
	float* x;
	float* y;
	float* z;
};

struct light
{
	positions pos;
	colors is;
	colors id;
};


struct spheres
{
	positions pos;
	colors color;
	float* radius;
	float* ka;
	float* kd;
	float* ks;
	int* alpha;

	__host__ __device__ bool hit_anything(const ray& r, double t_min, double t_max, hit_record& rec) const {

		hit_record temp_rec;
		bool hit_anything = false;
		auto closest_so_far = t_max;

		for (int i = 0; i < SPHERE_COUNT; ++i) {
			if (hit(i, r, t_min, closest_so_far, temp_rec)) {
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}

		return hit_anything;
	}

	__host__ __device__ bool hit(int i, const ray& r, double t_min, double t_max, hit_record& rec) const
	{
		float3 center = make_float3(pos.x[i], pos.y[i], pos.z[i]);

		float3 oc = r.origin - center; auto a = length_squared(r.direction);
		auto b = dot(oc, r.direction);
		auto c = length_squared(oc) - radius[i] * radius[i];

		auto discriminant = b * b - a * c;
		if (discriminant < 0)
			return false;

		auto sdet = sqrtf(discriminant);

		auto root = (-b - sdet) / a;
		if (root < t_min || t_max < root) {
			root = (-b + sdet) / a;
			if (root < t_min || t_max < root)
				return false;
		}

		rec.v = -1 * r.direction;
		rec.index = i;
		rec.t = root;
		rec.p = r.at(rec.t);
		rec.normal = (rec.p - center) / radius[i];

		return true;
	}
};

struct camera {

	float viewport_width;
	float viewport_height;
	float3 vup;
	float3 origin;
	float3 lower_left_corner;
	float3 horizontal;
	float3 vertical;
	float3 dir;
	float3 look_at;

	// neccesary default constructor
	camera() {}

	__host__ __device__ camera(float3 lookfrom, float3 lookat, float3 vup, float vfov, float aspect_ratio)
	{
		this->vup = vup;
		this->look_at = lookat;
		auto theta = degrees_to_radians(vfov);
		auto h = tan(theta / CAMERA_VIEWPORT);
		viewport_height = CAMERA_VIEWPORT * h;
		viewport_width = aspect_ratio * viewport_height;
		dir = lookfrom - lookat;
		auto w = normalize(dir);
		auto u = normalize(cross(vup, w));
		auto v = cross(w, u);

		origin = lookfrom;
		horizontal = viewport_width * u;
		vertical = viewport_height * v;
		lower_left_corner = origin - horizontal / 2 - vertical / 2 - w;
	}

	void resize_view(float aspect_ratio)
	{
		dir = origin - look_at;
		auto w = normalize(dir);
		auto u = normalize(cross(vup, w));
		auto v = cross(w, u);
		viewport_width = aspect_ratio * viewport_height;

		horizontal = viewport_width * u;
		vertical = viewport_height * v;
		lower_left_corner = origin - horizontal / 2 - vertical / 2 - w;
	}

	__host__ __device__ void move_look(float dx, float dy)
	{
		lookat(look_at + make_float3(dx, dy, 0));
	}

	void move_origin(float3 shift)
	{
		float3 viewDir = look_at - origin;
		origin = origin + shift;
		lookat(origin + viewDir);
		dir = origin - look_at;
	}

	__host__ __device__ void lookat(float3 pt)
	{
		auto w = normalize(origin - pt);
		auto u = normalize(cross(vup, w));
		auto v = cross(w, u);

		horizontal = viewport_width * u;
		vertical = viewport_height * v;
		lower_left_corner = origin - horizontal / 2 - vertical / 2 - w;
		this->look_at = pt;
	}

	__host__ __device__ ray get_ray(float s, float t) const {
		return ray(origin, lower_left_corner + s * horizontal + t * vertical - origin);
	}
};

struct fps_counter
{
	int n;
	float fpsSum;

	float avg_fps(float new_ms)
	{
		fpsSum += 1000.0f / new_ms;
		return fpsSum / ++n;
	}
};
