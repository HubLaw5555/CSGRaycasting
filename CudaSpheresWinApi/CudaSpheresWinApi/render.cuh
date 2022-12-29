#include "global.cuh"


__device__ __host__ unsigned int compute_int_color(const float3& v)
{
	unsigned int color = unsigned int(v.z * 255);
	color |= unsigned int(v.y * 255) << 8;
	color |= unsigned int(v.x * 255) << 16;
	return color;
}

__device__ __host__ float3 phong_light(const light& lights, const spheres& sphere, const hit_record& hit)
{
	int i = hit.index;
	float3 N = normalize(hit.normal);
	float3 V = normalize(hit.v);

	float ka = sphere.ka[i];
	float kd = sphere.kd[i];
	float ks = sphere.ks[i];
	int alpha = sphere.alpha[i];

	float3 sphereColor = make_float3(sphere.color.r[i], sphere.color.g[i], sphere.color.b[i]);
	float3 difuseColor = make_float3(0, 0, 0);
	float3 specularColor = make_float3(0, 0, 0);
	float3 pos = make_float3(sphere.pos.x[i], sphere.pos.y[i], sphere.pos.z[i]);

	for (int j = 0; j < LIGHT_COUNT; ++j)
	{
		float3 lightPosition = make_float3(lights.pos.x[j], lights.pos.y[j], lights.pos.z[j]);
		float3 L = normalize(lightPosition - pos);
		difuseColor = difuseColor + dot(L, N) * make_float3(lights.id.r[j], lights.id.g[j], lights.id.b[j]);

		float3 R = normalize(2.0f * dot(L, N) * N - L);
		float product = max(0.0f, dot(R, V));
		if (product > 0.0f)
		{
			product = fastPow(product, alpha);
		}

		specularColor = specularColor + product * make_float3(lights.is.r[j], lights.is.g[j], lights.is.b[j]);
	}

	float3 I = ka * make_float3(sphere.color.r[i], sphere.color.g[i], sphere.color.b[i]) + kd * difuseColor + ks * specularColor;
	I.x = clamp(I.x, .0f, 1.0f);
	I.y = clamp(I.y, .0f, 1.0f);
	I.z = clamp(I.z, .0f, 1.0f);

	return I;
}


__device__ __host__ float3 ray_color(const ray& r, const spheres& sphere, const light& lights) {
	hit_record rec;
	if (sphere.hit_anything(r, 0, 1000, rec))  // 1000 is infinity
	{
		return phong_light(lights, sphere, rec);
	}
	float3 unit_direction = normalize(r.direction);
	auto t = 0.5 * (unit_direction.y + 1.0);
	return (1.0 - t) * make_float3(1, 1, 1) + t * make_float3(0.5f, 0.7f, 1.0f);
}

__global__ void render(unsigned int* mem_ptr, spheres sphere, camera cam, light lights, int n, int maxX, int maxY)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= maxX || y >= maxY)
		return;

	// ANTiALLIASED try
	/*float3 c = make_float3(1, 1, 1);
	for (int i = 0; i < SAMPLES_PER_PIXEL; ++i)
	{
		ray r = cam.get_ray((float(x) + random_float()) / float(maxX - 1), (float(maxY - y) + random_float()) / float(maxY - 1));
		c = c + ray_color(r, spheres, lights);
	}
	c = c / float(SAMPLES_PER_PIXEL);*/
	ray r = cam.get_ray(float(x) / (maxX - 1), float(maxY - y) / (maxY - 1));
	float3 c = ray_color(r, sphere, lights);
	mem_ptr[y * maxX + x] = compute_int_color(c);
}

