#include "render.cuh"

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
void MouseMove(HWND hwnd, WPARAM wp, LPARAM lp);
void KeyPress(int keyCode);
void initBackBuffer(HWND hwnd);
void initialise_csg();
void initialize_memory();
void copy_scene_to_gpu(int n);
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

camera gpu_cam;
light gpu_lights, cpu_lights;
csg_scene cpu_scene, gpu_scene;

float curr_x = .0f, curr_y = .0f;
float prev_x = 0, prev_y = 0;

float3 shift = make_float3(0, 0, 0);

int reshaped_width = 0;
int reshaped_height = 0;
bool current_reshape = false;


// now here, after that in globals
const int sphere_cnt = 2;
const int lvls = 2;

void initialize_memory()
{
	cpu_buffer = new unsigned int[WIDTH * HEIGHT];
	memset(cpu_buffer, 0, sizeof(unsigned int) * WIDTH * HEIGHT);

	checkCudaErrors(cudaMalloc((void**)&cuda_buffer, WIDTH * HEIGHT * sizeof(unsigned int)));

	initialise_csg();
	cpu_scene.calculate_bounding_boxes();
	float x1 = cpu_scene.bounding.pos.x[0];
	float y1 = cpu_scene.bounding.pos.y[0];
	float z1 = cpu_scene.bounding.pos.z[0];
	float x2 = cpu_scene.bounding.pos.x[1];
	float y2 = cpu_scene.bounding.pos.y[1];
	float z2 = cpu_scene.bounding.pos.z[1];
	float x3 = cpu_scene.bounding.pos.x[2];
	float y3 = cpu_scene.bounding.pos.y[2];
	float z3 = cpu_scene.bounding.pos.z[2];

	float r1 = cpu_scene.bounding.radius[0];
	float r2 = cpu_scene.bounding.radius[1];
	float r3 = cpu_scene.bounding.radius[2];
	copy_scene_to_gpu(2);

	gpu_cam = camera(make_float3(0, 0.7f, 10.7f), make_float3(0, 0, -1.0f) /*make_float3(0,0,0), make_float3(0,0,1)*/,
		make_float3(0.0f, 1.0f, 0.0f), 30.0f, RATIO);
}

void draw(HWND hwnd)
{

	dim3 blocksCount = dim3(WIDTH, HEIGHT);
	//dim3 threadsCount = dim3(, framesY);
	float time;

	int bigPow = pow(2, lvls) - 1;
	int shared_size = 2 * bigPow * sizeof(float) + bigPow * sizeof(int) + sphere_cnt * sizeof(float3);
	render <<<blocksCount, 32, shared_size >>> (cuda_buffer, gpu_cam, gpu_lights, gpu_scene, WIDTH, HEIGHT);
	cudaDeviceSynchronize();
	cudaEventRecord(start_gpu_cpu, 0);
	checkCudaErrors(cudaMemcpy(cpu_buffer, cuda_buffer, WIDTH * HEIGHT * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	cudaEventRecord(stop_gpu_cpu, 0);
	cudaEventSynchronize(stop_render);
	cudaEventElapsedTime(&time, start_gpu_cpu, stop_gpu_cpu);
	time_gpu_cpu += time;

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

		windText += " ms     CPU -> GPU copy: " + std::to_string(time_cpu_gpu) +
			" ms     Avg GPU -> CPU copy: " + std::to_string(time_gpu_cpu / float(gpu_cpu_cnt)) + " ms";
		gpu_cpu_cnt++;

		SetWindowText(hwnd, windText.c_str());
	}

	cudaEventDestroy(start_render);
	cudaEventDestroy(stop_render);
	cudaEventDestroy(start_cpu_gpu);
	cudaEventDestroy(stop_cpu_gpu);
	cudaEventDestroy(start_gpu_cpu);
	cudaEventDestroy(stop_gpu_cpu);

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


void initialise_csg()
{
	int m = LIGHT_COUNT;

	int levels = 2;
	int n = pow(2, levels - 1);
	int nodes = pow(2, levels) - 1;
	//scene allocation
	cpu_scene = csg_scene(levels);

	checkCudaErrors(cudaMalloc((void**)&gpu_scene.objects.color.r, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.objects.color.g, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.objects.color.b, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.objects.primitives.pos.x, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.objects.primitives.pos.y, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.objects.primitives.pos.z, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.objects.primitives.radius, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.objects.ka, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.objects.kd, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.objects.ks, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.objects.alpha, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.bounding.pos.x, nodes * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.bounding.pos.y, nodes * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.bounding.pos.z, nodes * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.bounding.radius, nodes * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.csg,  nodes* sizeof(int)));
	/*checkCudaErrors(cudaMalloc((void**)&gpu_scene.levels, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.count, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&gpu_scene.nodesCount, sizeof(int)));*/


	// lights alocation
	cpu_lights.is.r = new float[m];
	cpu_lights.is.g = new float[m];
	cpu_lights.is.b = new float[m];
	cpu_lights.id.r = new float[m];
	cpu_lights.id.g = new float[m];
	cpu_lights.id.b = new float[m];
	cpu_lights.pos.x = new float[m];
	cpu_lights.pos.y = new float[m];
	cpu_lights.pos.z = new float[m];

	checkCudaErrors(cudaMalloc((void**)&gpu_lights.is.r, m * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_lights.is.g, m * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_lights.is.b, m * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_lights.id.r, m * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_lights.id.g, m * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_lights.id.b, m * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_lights.pos.x, m * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_lights.pos.y, m * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&gpu_lights.pos.z, m * sizeof(float)));

	for (int i = 0; i < m; ++i)
	{
		cpu_lights.is.r[i] = random_float(0.0f, 1.0f);
		cpu_lights.is.g[i] = random_float(0.0f, 1.0f);
		cpu_lights.is.b[i] = random_float(0.0f, 1.0f);
		cpu_lights.id.r[i] = random_float(0.0f, 1.0f);
		cpu_lights.id.g[i] = random_float(0.0f, 1.0f);
		cpu_lights.id.b[i] = random_float(0.0f, 1.0f);
		cpu_lights.pos.x[i] = SPREAD * RATIO * random_float(-1.0f, 1.0f);
		cpu_lights.pos.y[i] = SPREAD * random_float(-1.0f, 1.0f);
		cpu_lights.pos.z[i] = SPREAD * random_float(-1.0f, 1.0f);
	}

	for (int i = 0; i < n; ++i)
	{
		cpu_scene.objects.color.r[i] = random_float(0.0f, 1.0f);
		cpu_scene.objects.color.g[i] = random_float(0.0f, 1.0f);
		cpu_scene.objects.color.b[i] = random_float(0.0f, 1.0f);


		cpu_scene.objects.ka[i] = random_float(0.2f, 0.4f);
		cpu_scene.objects.kd[i] = random_float(0.0f, 0.03f);
		cpu_scene.objects.ks[i] = random_float(0.2f, 0.7f);
		cpu_scene.objects.alpha[i] = random_float(10, 100);
	}

	cpu_scene.objects.primitives.pos.x[0] =
		cpu_scene.objects.primitives.pos.y[0] =
		cpu_scene.objects.primitives.pos.z[0] = 0.0f;


	cpu_scene.objects.primitives.pos.x[1] = 
		cpu_scene.objects.primitives.pos.y[1] =
		cpu_scene.objects.primitives.pos.z[1] = 1.0f;

	cpu_scene.objects.primitives.radius[0] = 1.0f;
	cpu_scene.objects.primitives.radius[1] = 1.0f;

	cpu_scene.csg[0] = 2;
}

void copy_scene_to_gpu(int levels)
{
	int n = pow(2, levels - 1); // spheres count
	int nodes = pow(2, levels) - 1; // nodes count
	int m = LIGHT_COUNT;
	cudaEventRecord(start_cpu_gpu, 0);
	// lights copy
	checkCudaErrors(cudaMemcpy(gpu_lights.is.r, cpu_lights.is.r, m * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_lights.is.g, cpu_lights.is.g, m * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_lights.is.b, cpu_lights.is.b, m * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_lights.id.r, cpu_lights.id.r, m * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_lights.id.g, cpu_lights.id.g, m * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_lights.id.b, cpu_lights.id.b, m * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_lights.pos.x, cpu_lights.pos.x, m * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_lights.pos.y, cpu_lights.pos.y, m * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_lights.pos.z, cpu_lights.pos.z, m * sizeof(float), cudaMemcpyHostToDevice));

	//scene copy
	checkCudaErrors(cudaMemcpy(gpu_scene.objects.color.r, cpu_scene.objects.color.r, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_scene.objects.color.g, cpu_scene.objects.color.g, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_scene.objects.color.b, cpu_scene.objects.color.b, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_scene.objects.primitives.pos.x, cpu_scene.objects.primitives.pos.x, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_scene.objects.primitives.pos.y, cpu_scene.objects.primitives.pos.y, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_scene.objects.primitives.pos.z, cpu_scene.objects.primitives.pos.z, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_scene.objects.primitives.radius, cpu_scene.objects.primitives.radius, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_scene.objects.ka, cpu_scene.objects.ka, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_scene.objects.kd, cpu_scene.objects.kd, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_scene.objects.ks, cpu_scene.objects.ks, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_scene.objects.alpha, cpu_scene.objects.alpha, n * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_scene.bounding.pos.x, cpu_scene.bounding.pos.x, nodes * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_scene.bounding.pos.y, cpu_scene.bounding.pos.y, nodes * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_scene.bounding.pos.z, cpu_scene.bounding.pos.z, nodes * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_scene.bounding.radius, cpu_scene.bounding.radius, nodes * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_scene.csg, cpu_scene.csg, nodes * sizeof(int), cudaMemcpyHostToDevice));
	/*checkCudaErrors(cudaMemcpy(&gpu_scene.levels, &cpu_scene.levels, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&gpu_scene.count, &cpu_scene.count, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&gpu_scene.nodesCount, &cpu_scene.nodesCount, sizeof(int), cudaMemcpyHostToDevice));*/

	cudaEventRecord(stop_cpu_gpu, 0);
	cudaEventSynchronize(stop_render);
	cudaEventElapsedTime(&time_cpu_gpu, start_cpu_gpu, stop_cpu_gpu);
}