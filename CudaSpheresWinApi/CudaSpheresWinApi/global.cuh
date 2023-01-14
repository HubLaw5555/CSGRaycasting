#pragma once
#include "utilities.cuh"
#include <memory>


const float RATIO = 16.0f / 9.0f;
const int HEIGHT = 720;
const int WIDTH = RATIO * HEIGHT;

const int framesX = 16;
const int framesY = 16;

const int SPHERE_COUNT = 2000;
const int LIGHT_COUNT = 20;
const float SPREAD = 5.5f;
const int SAMPLES_PER_PIXEL = 5;
const float MOUSE_SPEED = 8.0f;
const float KEY_BATCH = 0.25f;
__device__ const float EPS = 1e-5;

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
	float3 range_l, range_r; // punkt przeciecia raya ze sfera
	float3 normal; // wektor normalny
	float3 v; // wektor do kamery
	float t1, t2; // parametr t raya dla ktorego sie przecina z kula
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
	float* radius;

	__host__ __device__ bool hit_anything(const ray& r, double t_min, double t_max, hit_record& rec) const
	{
		hit_record temp_rec;
		bool hit_anything = false;
		//auto closest_so_far = t_max;

		for (int i = 0; i < SPHERE_COUNT; ++i) {
			if (hit(i, r, t_min, /*closest_so_far*/t_max, temp_rec)) {
				hit_anything = true;
				//closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}

		return hit_anything;
	}

	__host__ __device__ bool hit(int i, const ray& r, double t_min, double t_max, hit_record& rec) const
	{
		float3 center = make_float3(pos.x[i], pos.y[i], pos.z[i]);
		float3 oc = r.origin - center;

		auto a = length_squared(r.direction);
		auto b = dot(oc, r.direction);
		auto c = length_squared(oc) - radius[i] * radius[i];

		auto discriminant = b * b - a * c;
		if (discriminant < -EPS)
			return false;

		if(discriminant >= EPS)
		{
			auto sdet = sqrtf(discriminant);

			// always root1 >= root2
			auto root1 = (-b - sdet) / a;
			auto root2 = (-b + sdet) / a;

			// jesli root1 < root2
			// if(root1 > root2)
			// {
			// 	auto temp = root2;
			// 	root2 = root1;
			// 	root1 = temp;
			// }

			if ((root1 < t_min && root2 < t_min) || (root1 > t_max && root2 > t_max))
				return false;

			/*if (root1 < t_min)
				root1 = t_min + EPS;
			if (root2 > t_max)
				root2 = t_max;*/
			if (root2 < t_min || root2 > t_max)
				return false;

			// if (root < t_min || t_max < root) {
			// 	root = (-b + sdet) / a;
			// 	if (root < t_min || t_max < root)
			// 		return false;

			rec.t1 = root1;
			rec.t2 = root2;
		}
		else
		{
			rec.t1 = -b / a - EPS;
			rec.t2 = -b / a + EPS;
		}

		rec.range_l = r.at(rec.t1);
		rec.range_r = r.at(rec.t2);

		rec.v = -1 * r.direction;
		rec.index = i;
		rec.normal = (rec.range_l - center) / radius[i];

		return true;
	}
};


struct scene_objects
{
	spheres primitives;
	colors color;
	float* ka;
	float* kd;
	float* ks;
	int* alpha;

	void allocate(int n)
	{
		primitives.pos.x = new float[n];
		primitives.pos.y = new float[n];
		primitives.pos.z = new float[n];
		primitives.radius = new float[n];

		color.r = new float[n];
		color.g = new float[n];
		color.b = new float[n];
		ka = new float[n];
		kd = new float[n];
		ks = new float[n];
		alpha = new int[n];
	}
};

struct csg_scene
{
	int levels;
	int nodesCount;
	int count; // spheres count

	// for 	leaves indices from sph struct
	// for other nodes set operations: 0 intersection, 1 sum, 2 difference left\right, 3 difference right\left
	// empty nodes (not in tree) -1
	int* csg;
	scene_objects objects;

	// bounding spheres
	spheres bounding;

	csg_scene() {}

	csg_scene(int n)
	{
		levels = n;
		evaluate_nodes_count();

		csg = new int[nodesCount];
		memset(csg, -1, nodesCount);

		objects.allocate(nodesCount);

		bounding.pos.x = new float[nodesCount];
		bounding.pos.y = new float[nodesCount];
		bounding.pos.z = new float[nodesCount];
		bounding.radius = new float[nodesCount];
	}

	void evaluate_nodes_count()
	{
		nodesCount = pow(2, levels) - 1;
		count = pow(2, levels - 1);
	}

	void calculate_bounding_boxes()
	{
		// copy spheres to leaves
		int index = 0;
		for (int i = pow(2, levels - 1) - 1; i < pow(2, levels) - 1; ++i)
		{
			csg[i] = index;
			float x = objects.primitives.pos.x[index];
			float y = objects.primitives.pos.y[index];
			float z = objects.primitives.pos.z[index];
			bounding.pos.x[i] = x;
			bounding.pos.y[i] = y;
			bounding.pos.z[i] = z;

			bounding.radius[i] = objects.primitives.radius[index++];
		}

		//propagate leaves upwards
		for (int i = levels - 1; i > 0; --i)
		{
			for (int j = pow(2, i - 1) - 1; j < pow(2, i) - 1; ++j)
			{
				int left = 2 * j + 1, right = 2 * j + 2;

				// empty node
				if (csg[left] == -1 || csg[right] == -1)
					continue;

				float r1 = bounding.radius[left], r2 = bounding.radius[right];
				float x1 = bounding.pos.x[left], y1 = bounding.pos.y[left], z1 = bounding.pos.z[left];
				float x2 = bounding.pos.x[right], y2 = bounding.pos.y[right], z2 = bounding.pos.z[right];

				switch (csg[j])
				{
					// intersection
					case 0:
					{
						bounding.radius[j] = min(r1, r2);
						bounding.pos.x[j] = r1 < r2 ? x1 : x2;
						bounding.pos.y[j] = r1 < r2 ? y1 : y2;
						bounding.pos.z[j] = r1 < r2 ? z1 : z2;
					}
					break;
					// sum
					case 1:
					{
						float r_dist = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
						float r = (r1 + r2 + r_dist) / 2.0f;
						float dir_x = (x2 - x1) / r_dist;
						float dir_y = (y2 - y1) / r_dist;
						float dir_z = (z2 - z1) / r_dist;
						bounding.pos.x[j] = x1 + (r - r1) * dir_x;
						bounding.pos.y[j] = y1 + (r - r1) * dir_y;
						bounding.pos.z[j] = z1 + (r - r1) * dir_z;
						bounding.radius[j] = r;
					}
					break;
					// difference left \ right
					case 2:
					{
						bounding.radius[j] = r1;
						bounding.pos.x[j] = x1;
						bounding.pos.y[j] = y1;
						bounding.pos.z[j] = z1;
					}
					break;
					// difference right \ left
					case 3:
					{
						bounding.radius[j] = r2;
						bounding.pos.x[j] = x2;
						bounding.pos.y[j] = y2;
						bounding.pos.z[j] = z2;
					}
					break;
					// -1 and err
					default:
						break;
				}
			}
		}
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
