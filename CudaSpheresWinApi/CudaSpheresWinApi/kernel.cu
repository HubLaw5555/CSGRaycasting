#include "render.cuh"

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
void MouseMove(HWND hwnd, WPARAM wp, LPARAM lp);
void KeyPress(int keyCode);
void initBackBuffer(HWND hwnd);
void init_spheres();
void free_memory();
void initialize_memory();
void move_camera(HWND hwnd);

HDC hBackDC = NULL;
HBITMAP hBackBitmap = NULL;


cudaEvent_t start_render, stop_render;
cudaEvent_t start_gpu_cpu, stop_gpu_cpu;
cudaEvent_t start_cpu_gpu, stop_cpu_gpu;

float time_render = .0f, time_cpu_gpu = .0f, time_gpu_cpu = .0f;
int gpu_cpu_cnt = 1;


unsigned int* cuda_buffer;
unsigned int* cpu_buffer;
spheres gpu_spheres, s;
camera gpu_cam;
light gpu_lights, l;

float curr_x = .0f, curr_y = .0f;
float prev_x = 0, prev_y = 0;

float3 shift = make_float3(0, 0, 0);

int reshaped_width = 0;
int reshaped_height = 0;
bool current_reshape = false;

void initialize_memory()
{
	cpu_buffer = new unsigned int[WIDTH * HEIGHT];
	memset(cpu_buffer, 0, sizeof(unsigned int) * WIDTH * HEIGHT);

	checkCudaErrors(cudaMalloc((void**)&cuda_buffer, WIDTH * HEIGHT * sizeof(unsigned int)));

	init_spheres();

	gpu_cam = camera(make_float3(0, 0.7f, 10.7f), make_float3(0, 0, -1.0f) /*make_float3(0,0,0), make_float3(0,0,1)*/,
		make_float3(0.0f, 1.0f, 0.0f), 30.0f, RATIO);
}


void render_cpu()
{
	for (int x = 0; x < WIDTH; ++x)
	{
		for (int y = 0; y < HEIGHT; ++y)
		{
			// ANTYALIASING
			/*float3 c = make_float3(0, 0, 0);
			ray r;
			for (int i = 0; i < SAMPLES_PER_PIXEL; ++i)
			{
				r = gpu_cam.get_ray((float(x) + random_float()) / (WIDTH - 1), (float(HEIGHT - y) + random_float()) / (HEIGHT - 1));
				c = c + ray_color(r, s, l);
			}
			c = c/SAMPLES_PER_PIXEL*/

			ray r = gpu_cam.get_ray(float(x) / (WIDTH - 1), float(y) / (HEIGHT - 1));
			float3 c = ray_color(r, s, l);

			cpu_buffer[y * WIDTH + x] = compute_int_color(c);
		}
	}
}

void draw(HWND hwnd)
{

	dim3 blocksCount = dim3(WIDTH / framesX, HEIGHT / framesY);
	dim3 threadsCount = dim3(framesX, framesY);
	float time;

	if (GPU_RENDER)
	{
		render <<<blocksCount, threadsCount >> > (cuda_buffer, gpu_spheres, gpu_cam, gpu_lights, SPHERE_COUNT, WIDTH, HEIGHT);
		cudaDeviceSynchronize();
		cudaEventRecord(start_gpu_cpu, 0);
		checkCudaErrors(cudaMemcpy(cpu_buffer, cuda_buffer, WIDTH * HEIGHT * sizeof(unsigned int), cudaMemcpyDeviceToHost));
		cudaEventRecord(stop_gpu_cpu, 0);
		cudaEventSynchronize(stop_render);
		cudaEventElapsedTime(&time, start_gpu_cpu, stop_gpu_cpu);
		time_gpu_cpu += time;
	}
	else
	{
		render_cpu();
	}

	SetBitmapBits(hBackBitmap, HEIGHT * WIDTH * sizeof(unsigned int), (const void*)(cpu_buffer));
	BitBlt(GetDC(hwnd), 0, 0, WIDTH, HEIGHT, hBackDC, 0, 0, SRCCOPY);
}

int WINAPI wWinMain(HINSTANCE hInstace, HINSTANCE hPrevInstace, LPWSTR lpCmdLine, int nCmdShow)
{

	MSG msg = { 0 };
	WNDCLASS wnd = { 0 };

	cudaEventCreate(&start_render);
	cudaEventCreate(&stop_render);
	cudaEventCreate(&start_cpu_gpu);
	cudaEventCreate(&stop_cpu_gpu);
	cudaEventCreate(&start_gpu_cpu);
	cudaEventCreate(&stop_gpu_cpu);

	wnd.lpfnWndProc = WndProc;
	wnd.hInstance = hInstace;
	wnd.lpszClassName = "Window";

	if (!RegisterClass(&wnd)) {
		return 0;
	}

	checkCudaErrors(cudaSetDevice(0));
	initialize_memory();


	HWND hwnd = CreateWindowEx(WS_EX_CLIENTEDGE, wnd.lpszClassName, "Window",
		WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, WIDTH, HEIGHT, NULL, NULL, hInstace, NULL);

	if (!hwnd) {
		return 0;
	}

	ShowWindow(hwnd, nCmdShow);
	UpdateWindow(hwnd);
	fps_counter fps;
	fps.n = 0;
	fps.fpsSum = .0f;

	while (true)
	{
		cudaEventRecord(start_render, 0);

		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			if (msg.message == WM_QUIT) {
				break;
			}

			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		move_camera(hwnd);
		draw(hwnd);

		cudaEventRecord(stop_render, 0);
		cudaEventSynchronize(stop_render);
		cudaEventElapsedTime(&time_render, start_render, stop_render);
		std::string windText = " Average fps: " + std::to_string(fps.avg_fps(time_render));

		if (GPU_RENDER)
		{
			windText += " ms     CPU -> GPU copy: " + std::to_string(time_cpu_gpu) + 
				" ms     Avg GPU -> CPU copy: " + std::to_string(time_gpu_cpu / float(gpu_cpu_cnt)) + " ms";
			gpu_cpu_cnt++;
		}

		SetWindowText(hwnd, windText.c_str());
	}

	cudaEventDestroy(start_render);
	cudaEventDestroy(stop_render);
	cudaEventDestroy(start_cpu_gpu);
	cudaEventDestroy(stop_cpu_gpu);
	cudaEventDestroy(start_gpu_cpu);
	cudaEventDestroy(stop_gpu_cpu);

	free_memory();

	return msg.wParam;
}


LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {

	wchar_t buffer[256];
	switch (msg) {
	case WM_CREATE:
		initBackBuffer(hwnd);
		break;
	case WM_DESTROY:
		DeleteDC(hBackDC);
		DeleteObject(hBackBitmap);
		PostQuitMessage(0);
		break;
	case WM_KEYDOWN:
		swprintf_s(buffer, 256, L"WM_KEYDOWN: 0x%x\n", wParam);
		KeyPress(wParam);
		OutputDebugStringW(buffer);
		break;

	case WM_KEYUP:
		swprintf_s(buffer, 256, L"WM_KEYUP: 0x%x\n", wParam);
		OutputDebugStringW(buffer);
		break;
	case WM_MOUSEMOVE:
		MouseMove(hwnd, wParam, lParam);
		break;
	}
	return DefWindowProc(hwnd, msg, wParam, lParam);
}

void KeyPress(int keyCode)
{
	switch (keyCode)
	{
	case 32: // space
		shift = shift + KEY_BATCH * (gpu_cam.vertical / gpu_cam.viewport_height);
		break;
	case 16: // shift
		shift = shift - KEY_BATCH * (gpu_cam.vertical / gpu_cam.viewport_height);
		break;
	case 0x57: // W
		shift = shift + KEY_BATCH * normalize(gpu_cam.look_at - gpu_cam.origin);
		break;
	case 0x53: // S
		shift = shift - KEY_BATCH * normalize(gpu_cam.look_at - gpu_cam.origin);
		break;
	case 0x41: // A
		shift = shift - KEY_BATCH * (gpu_cam.horizontal / gpu_cam.viewport_width);
		break;
	case 0x44: // D
		shift = shift + KEY_BATCH * (gpu_cam.horizontal / gpu_cam.viewport_width);
		break;
	}
}


void MouseMove(HWND hwnd, WPARAM wp, LPARAM lp)
{
	float x = LOWORD(lp);
	float y = HIWORD(lp);

	// to [-1,1]
	curr_x = 2.0f * x / WIDTH - 1.0f;
	curr_y = 2.0f * (HEIGHT - y) / HEIGHT - 1.0f;
}


void move_camera(HWND hwnd)
{
	gpu_cam.move_origin(shift);
	gpu_cam.move_look(MOUSE_SPEED * (curr_x - prev_x), MOUSE_SPEED * (curr_y - prev_y));


	prev_x = curr_x;
	prev_y = curr_y;
	shift = make_float3(0, 0, 0);
}

void initBackBuffer(HWND hwnd) {
	HDC hWinDC = GetDC(hwnd);
	BITMAPINFO bmi = { 0 };
	bmi.bmiHeader.biSize = sizeof(BITMAPCOREHEADER);
	bmi.bmiHeader.biWidth = WIDTH;
	bmi.bmiHeader.biHeight = -HEIGHT;
	bmi.bmiHeader.biPlanes = 1;
	bmi.bmiHeader.biBitCount = 32;
	bmi.bmiHeader.biCompression = BI_RGB;

	hBackDC = CreateCompatibleDC(hWinDC);
	hBackBitmap = CreateCompatibleBitmap(hWinDC, WIDTH, HEIGHT);
	SetBitmapBits(hBackBitmap, HEIGHT * WIDTH * sizeof(unsigned int), (const void*)(cpu_buffer));

	SelectObject(hBackDC, hBackBitmap);
	ReleaseDC(hwnd, hWinDC);
}


void init_spheres()
{
	int n = SPHERE_COUNT;
	int m = LIGHT_COUNT;

	s.color.r = new float[n];
	s.color.g = new float[n];
	s.color.b = new float[n];

	checkCudaErrors(cudaMalloc((void**)&gpu_spheres.color.r, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_spheres.color.g, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_spheres.color.b, n * sizeof(float)));

	s.pos.x = new float[n];
	s.pos.y = new float[n];
	s.pos.z = new float[n];

	checkCudaErrors(cudaMalloc((void**)&gpu_spheres.pos.x, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_spheres.pos.y, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_spheres.pos.z, n * sizeof(float)));

	s.radius = new float[n];
	checkCudaErrors(cudaMalloc((void**)&gpu_spheres.radius, n * sizeof(float)));

	s.ka = new float[n];
	s.kd = new float[n];
	s.ks = new float[n];
	s.alpha = new int[n];
	checkCudaErrors(cudaMalloc((void**)&gpu_spheres.ka, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_spheres.kd, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_spheres.ks, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_spheres.alpha, n * sizeof(float)));



	l.is.r = new float[m];
	l.is.g = new float[m];
	l.is.b = new float[m];
	l.id.r = new float[m];
	l.id.g = new float[m];
	l.id.b = new float[m];

	l.pos.x = new float[m];
	l.pos.y = new float[m];
	l.pos.z = new float[m];

	checkCudaErrors(cudaMalloc((void**)&gpu_lights.is.r, m * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_lights.is.g, m * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_lights.is.b, m * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_lights.id.r, m * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_lights.id.g, m * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_lights.id.b, m * sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&gpu_lights.pos.x, m * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_lights.pos.y, m * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_lights.pos.z, m * sizeof(float)));


	for (int i = 0; i < n; ++i)
	{
		s.color.r[i] = random_float(0, 1);
		s.color.g[i] = random_float(0, 1);
		s.color.b[i] = random_float(0, 1);


		s.pos.x[i] = SPREAD * RATIO * random_float(-1, 1);
		s.pos.y[i] = SPREAD * random_float(-1, 1);
		s.pos.z[i] = SPREAD * random_float(-1, 1);

		s.radius[i] = 0.05f + random_float(0, 1) * 15.0f / 100.0f;

		s.ka[i] = random_float(0.2f, 0.4f);
		s.kd[i] = random_float(0.0f, 0.03f);
		s.ks[i] = random_float(0.2f, 0.7f);
		s.alpha[i] = random_float(10, 100);
	}

	for (int i = 0; i < m; ++i)
	{
		l.is.r[i] = random_float(0.0f, 1.0f);
		l.is.g[i] = random_float(0.0f, 1.0f);
		l.is.b[i] = random_float(0.0f, 1.0f);
		l.id.r[i] = random_float(0.0f, 1.0f);
		l.id.g[i] = random_float(0.0f, 1.0f);
		l.id.b[i] = random_float(0.0f, 1.0f);

		l.pos.x[i] = SPREAD * RATIO * random_float(-1.0f, 1.0f);
		l.pos.y[i] = SPREAD * random_float(-1.0f, 1.0f);
		l.pos.z[i] = SPREAD * random_float(-1.0f, 1.0f);
	}

	cudaEventRecord(start_cpu_gpu, 0);

	checkCudaErrors(cudaMemcpy(gpu_spheres.color.r, s.color.r, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_spheres.color.g, s.color.g, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_spheres.color.b, s.color.b, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_spheres.radius, s.radius, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_spheres.ka, s.ka, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_spheres.kd, s.kd, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_spheres.ks, s.ks, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_spheres.alpha, s.alpha, n * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_spheres.pos.x, s.pos.x, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_spheres.pos.y, s.pos.y, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_spheres.pos.z, s.pos.z, n * sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(gpu_lights.is.r, l.is.r, m * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_lights.is.g, l.is.g, m * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_lights.is.b, l.is.b, m * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_lights.id.r, l.id.r, m * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_lights.id.g, l.id.g, m * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_lights.id.b, l.id.b, m * sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(gpu_lights.pos.x, l.pos.x, m * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_lights.pos.y, l.pos.y, m * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_lights.pos.z, l.pos.z, m * sizeof(float), cudaMemcpyHostToDevice));


	cudaEventRecord(stop_cpu_gpu, 0);
	cudaEventSynchronize(stop_render);
	cudaEventElapsedTime(&time_cpu_gpu, start_cpu_gpu, stop_cpu_gpu);
}

void free_memory()
{
	delete [] s.color.r;
	delete [] s.color.g;
	delete [] s.color.b;
	delete [] s.pos.x;
	delete [] s.pos.y;
	delete [] s.pos.z;
	delete [] s.radius;
	delete [] s.alpha;
	delete [] s.kd;
	delete [] s.ks;
	delete [] l.is.r;
	delete [] l.is.g;
	delete [] l.is.b;
	delete [] l.id.r;
	delete [] l.id.g;
	delete [] l.id.b;
	delete [] l.pos.x;
	delete [] l.pos.y;
	delete [] l.pos.z;

	checkCudaErrors(cudaFree(gpu_spheres.color.r));
	checkCudaErrors(cudaFree(gpu_spheres.color.g));
	checkCudaErrors(cudaFree(gpu_spheres.color.b));
	checkCudaErrors(cudaFree(gpu_spheres.pos.x));
	checkCudaErrors(cudaFree(gpu_spheres.pos.y));
	checkCudaErrors(cudaFree(gpu_spheres.pos.z));
	checkCudaErrors(cudaFree(gpu_spheres.radius));
	checkCudaErrors(cudaFree(gpu_spheres.alpha));
	checkCudaErrors(cudaFree(gpu_spheres.kd));
	checkCudaErrors(cudaFree(gpu_spheres.ks));
	checkCudaErrors(cudaFree(gpu_lights.is.r));
	checkCudaErrors(cudaFree(gpu_lights.is.g));
	checkCudaErrors(cudaFree(gpu_lights.is.b));
	checkCudaErrors(cudaFree(gpu_lights.id.r));
	checkCudaErrors(cudaFree(gpu_lights.id.g));
	checkCudaErrors(cudaFree(gpu_lights.id.b));
	checkCudaErrors(cudaFree(gpu_lights.pos.x));
	checkCudaErrors(cudaFree(gpu_lights.pos.y));
	checkCudaErrors(cudaFree(gpu_lights.pos.z));
}