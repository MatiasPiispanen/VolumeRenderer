#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <SDL/SDL.h>

#define PI 3.14159265

typedef short int16_t;
typedef unsigned short uint16_t;

SDL_Surface *surf;
SDL_Event keyevent;
bool running = true;
int height = 480, width = 640, camera_dist = 250;
char *odata;
dim3 block_size, grid_size;
void *d = NULL, *result = NULL;
unsigned int work_dim[3] = {256, 256, 113};
float sample_rate = 1.0f;
uint16_t maximum = 0;
float3 scale_data;
float3 rotation;

__constant__ float err = 1.0f;

using namespace std;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
		cudaError_t _m_cudaStat = value;										\
		if (_m_cudaStat != cudaSuccess) {										\
			fprintf(stderr, "Error %s at line %d in file %s\n",					\
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
					exit(1);															\
		} }

__device__ uint16_t bytereverse(uint16_t number) {
	number = (number << 8) | (number >> 8);
	return number;
}

__device__ void normalize(float4* vec) {
	float len;
	len = sqrt(pow(vec->x, 2) + pow(vec->y, 2) + pow(vec->z, 2));
	vec->x = vec->x / len;
	vec->y = vec->y / len;
	vec->z = vec->z / len;
}

__device__ float dist(float4 *src, float4 *dst) {
	return sqrt( pow((*src).x - (*dst).x, 2) +  pow((*src).y - (*dst).y, 2) + pow((*src).z - (*dst).z, 2));
}

__device__ float dot(float4 *first, float4 *second) {
	return (((*first).x * (*second).x)) + ((*first).y * (*second).y) + ((*first).z * (*second).z);
}

__device__ void vecAdd(float4 *vec, float4 *add) {
	vec->x += add->x;
	vec->y += add->y;
	vec->z += add->z;
}

__device__ void vecSubtract(float4 *vec, float4 *sub) {
	vec->x -= sub->x;
	vec->y -= sub->y;
	vec->z -= sub->z;
}

__device__ void vecAddColor(float4 *vec, float4 *add) {
	vecAdd(vec, add);
	vec->w += add->w;
}

__device__ float4 vecAddColor(float4 vec, float4 add) {
	float4 temp = vec;

	vecAdd(&temp, &add);
	temp.w += add.w;

	return temp;
}

__device__ void matrixMul(float* mat) {
	int dim = 4;
	float cmat[4][4];
	for(int i = 0; i < dim; i++) {
		for(int j = 0; j < dim; j++) {
			cmat[i][j] = mat[i*dim+j];
			mat[i*dim+j] = 0;
		}
	}


	for(int i = 0; i < dim; i++) {
		for(int j = 0; j < dim; j++) {
			for(int k = 0; k < dim; k++) {
				mat[i*dim+j] += cmat[i][k] * cmat[k][j];
			}
		}
	}
}

__device__ void transform(float4 *vec, float *mat) {
	int dim = 4;
	float cmat[4][4];
	float temp[4] = { vec->x, vec->y, vec->z, vec->w };
	*vec = make_float4(0.f, 0.f, 0.f, 0.f);

	for(int i = 0; i < dim; i++) {
		for(int j = 0; j < dim; j++) {
			cmat[i][j] = mat[i*dim+j];
		}
	}

	for(int i = 0; i < dim; i++) {
		vec->x += cmat[0][i] * temp[i];
		vec->y += cmat[1][i] * temp[i];
		vec->z += cmat[2][i] * temp[i];
		vec->w += cmat[3][i] * temp[i];
	}
}

__device__ void rotate(float4 *vec, float3 *rotation) {
	float xrot[4][4] = {
			{1.f, 0.f, 0.f, 0.f},
			{0.f, cosf(rotation->x), (-1)*sinf(rotation->x), 0.f},
			{0.f, sinf(rotation->x), cosf(rotation->x), 0.f},
			{0.f, 0.f, 0.f, 1.f}
	};

	float yrot[4][4] = {
			{cosf(rotation->y), 0.f, (-1)*sinf(rotation->y), 0.f},
			{0.f, 1.f, 0.f, 0.f},
			{sinf(rotation->y), 0.f, cosf(rotation->y), 0.f},
			{0.f, 0.f, 0.f, 1.f}
	};

	float zrot[4][4] = {
			{cosf(rotation->z), (-1)*sinf(rotation->z), 0.f, 0.f},
			{sinf(rotation->z), cosf(rotation->z), 0.f, 0.f},
			{0.f, 0.f, 1.f, 0.f},
			{0.f, 0.f, 0.f, 1.f}
	};

	transform(vec, (float*)xrot);
	transform(vec, (float*)yrot);
	transform(vec, (float*)zrot);
}

__device__ float rayPlaneIntersection(float4 *camera_pos, float4 *ray, float4 *plane_point, float4 *normal) {
	float4 w = make_float4((*plane_point).x - (*camera_pos).x, (*plane_point).y - (*camera_pos).y, (*plane_point).z - (*camera_pos).z, 1.f);

	if(dot(normal, ray) != 0) {
		return dot(normal, &w) / dot(normal, ray);
	}
	else {
		// Magic number = "max float"
		return 999999.9;
	}
}

__device__ void scale(float4 *vec, float s) {
	vec->x *= s;
	vec->y *= s;
	vec->z *= s;
}

__device__ void scaleColor(float4 *vec, float s) {
	scale(vec, s);
	vec->w *= s;
}

__device__ float4 scaleColor(float4 vec, float s) {
	float4 temp = vec;

	scale(&temp, s);
	temp.w *= s;

	return temp;
}

__device__ int index(unsigned int x, unsigned int y, unsigned int z, int3 dim) {
	//FIXME: Not sure if I got this right
	return (z * (dim.x) * (dim.y)) + (y * (dim.x)) + x;
}

__device__ float interpolate(uint16_t *field, float4 *vec, int3 dim, uint16_t maximum) {
	int i0, j0, k0, i1, j1, k1;
	float sx0, sx1, sy0, sy1, sz0, sz1, v0, v1;

	if(vec->x < 0.5f) {
		vec->x = 0.5f;
	}

	if(vec->y < 0.5f) {
		vec->y = 0.5f;
	}

	if(vec->z < 0.5f) {
		vec->z = 0.5f;
	}

	if(vec->x > dim.x - 0.5f) {
		vec->x = dim.x - 0.5f;
	}

	if(vec->y > dim.y - 0.5f) {
		vec->y = dim.y - 0.5f;
	}

	if(vec->z > dim.z - 0.5f) {
		vec->z = dim.z - 0.5f;
	}

	i0 = (int)vec->x;
	i1 = i0+1;

	j0 = (int)vec->y;
	j1 = j0+1;

	k0 = (int)vec->z;
	k1 = k0+1;

	sx1 = vec->x -i0;
	sx0 = 1-sx1;

	sy1 = vec->y -j0;
	sy0 = 1-sy1;

	sz1 = vec->z -k0;
	sz0 = 1-sz1;

	v0 = sx0 * (sy0 * (float)field[index(i0,j0,k0,dim)] + sy1 * (float)field[index(i0,j1,k0,dim)]) +
		 sx1 * (sy0 * (float)field[index(i1,j0,k0,dim)] + sy1 * (float)field[index(i1,j1,k0,dim)]);

	v1 = sx0 * (sy0 * (float)field[index(i0,j0,k1,dim)] + sy1 * (float)field[index(i0,j1,k1,dim)]) +
		 sx1 * (sy0 * (float)field[index(i1,j0,k1,dim)] + sy1 * (float)field[index(i1,j1,k1,dim)]);

	return (sz0*v0 + sz1*v1) / maximum; // normalize the result
}

/**
 * CUDA kernel function that reverses the order of bytes in the array.
 */
__global__ void bytereverse(void *data, int len) {
	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint16_t *idata = (uint16_t*) data;

	if(x < len) {
		idata[x] = bytereverse(idata[x]);
	}
}

/**
 * CUDA kernel function for volume ray marching of CT scan data
 */
__global__ void volumeRayMarching(void* data, void *result, unsigned int height, unsigned int width, unsigned int camera_dist,
		float box_width, float box_height, float box_depth, float sample_rate, uint16_t maximum, float3 scale_data, float3 rotation) {
	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	char *odata = (char*)result;
	uint16_t *idata = (uint16_t*)data;

	if(x < width && y < height) {
		/* Distance and index of the nearest intersection with the bounding cube. Indices in the order of
		 * front, back, left, right, top, bottom. front = side of the camera in the initial position.
		 */
		float closest_wall_dist = -1.f;
		float temp_dist = -1.f;

		/*float4 dir = make_float4((float)((int)x-((int)width/2)), (float)((int)y-((int)height/2)), (float)(camera_dist), 1.f);*/
		float4 dir = make_float4((float)((int)x-((int)width/2)), (float)((int)y-((int)height/2)), (float)((int)camera_dist)*(-1)+100.f, 0.f);
		float4 camera_pos = make_float4(0.f, 0.f, (float)((int)camera_dist)*(-1), 1.f);
		float4 vec;

		vecSubtract(&dir, &camera_pos);
		rotate(&dir, &rotation);
		rotate(&camera_pos, &rotation);

		normalize(&dir);


		// Check intersections with bounding box walls and find the nearest one
		// FIXME: Wow, horrible code
		float4 normal = make_float4(0.f, 0.f, -1.f, 1.f);
		float4 plane_point = make_float4(0.f, 0.f, (-1)*(float)box_depth/2, 1.f);

		if(dir.z * normal.z < 0) {
			temp_dist = rayPlaneIntersection(&camera_pos, &dir, &plane_point, &normal);
			vec = make_float4(temp_dist*dir.x, temp_dist*dir.y, temp_dist*dir.z, 1.f);

			if(abs(vec.x) <= (float)box_width/2 && abs(vec.y) <= (float)box_height/2) {
				if(closest_wall_dist < 0 || (temp_dist > 0 && temp_dist < closest_wall_dist)) {
					closest_wall_dist = temp_dist;
				}
			}
		}

		plane_point.z *= -1;
		normal.z *= -1;
		if(dir.z * normal.z < 0) {
			temp_dist = rayPlaneIntersection(&camera_pos, &dir, &plane_point, &normal);
			vec = make_float4(temp_dist*dir.x, temp_dist*dir.y, temp_dist*dir.z, 1.f);

			if(abs(vec.x) <= (float)box_width/2 && abs(vec.y) <= (float)box_height/2) {
				if(closest_wall_dist < 0 || (temp_dist > 0 && temp_dist < closest_wall_dist)) {
					closest_wall_dist = temp_dist;
				}
			}
		}

		plane_point.z = 0.f;
		plane_point.x = (float)box_width/2;
		normal.z = 0.f;
		normal.x = 1.f;

		if(dir.x * normal.x < 0) {
			temp_dist = rayPlaneIntersection(&camera_pos, &dir, &plane_point, &normal);
			vec = make_float4(temp_dist*dir.x, temp_dist*dir.y, temp_dist*dir.z, 1.f);

			if(abs(vec.z) <= (float)box_depth/2 && abs(vec.y) <= (float)box_height/2) {
				if(closest_wall_dist < 0 || (temp_dist > 0 && temp_dist < closest_wall_dist)) {
					closest_wall_dist = temp_dist;
				}
			}
		}

		plane_point.x *= -1;
		normal.x *= -1;

		if(dir.x * normal.x < 0) {
			temp_dist = rayPlaneIntersection(&camera_pos, &dir, &plane_point, &normal);
			vec = make_float4(temp_dist*dir.x, temp_dist*dir.y, temp_dist*dir.z, 1.f);

			if(abs(vec.z) <= (float)box_depth/2 && abs(vec.y) <= (float)box_height/2) {
				if(closest_wall_dist < 0 || (temp_dist > 0 && temp_dist < closest_wall_dist)) {
					closest_wall_dist = temp_dist;
				}
			}
		}

		plane_point.x = 0.f;
		plane_point.y = (float)box_height/2;
		normal.x = 0.f;
		normal.y = 1.f;

		if(dir.y * normal.y < 0) {
			temp_dist = rayPlaneIntersection(&camera_pos, &dir, &plane_point, &normal);
			vec = make_float4(temp_dist*dir.x, temp_dist*dir.y, temp_dist*dir.z, 1.f);

			if(abs(vec.z) <= (float)box_depth/2 && abs(vec.x) <= (float)box_width/2) {
				if(closest_wall_dist < 0 || (temp_dist > 0 && temp_dist < closest_wall_dist)) {
					closest_wall_dist = temp_dist;
				}
			}
		}

		plane_point.y *= -1;
		normal.y *= -1;

		if(dir.y * normal.y < 0) {
			temp_dist = rayPlaneIntersection(&camera_pos, &dir, &plane_point, &normal);
			vec = make_float4(temp_dist*dir.x, temp_dist*dir.y, temp_dist*dir.z, 1.f);

			if(abs(vec.z) <= (float)box_depth/2 && abs(vec.x) <= (float)box_width/2) {
				if(closest_wall_dist < 0 || (temp_dist > 0 && temp_dist < closest_wall_dist)) {
					closest_wall_dist = temp_dist;
				}
			}
		}

		// Start volume ray marching only if there was an intersection with the bounding box.
		if(!(closest_wall_dist <= 0)) {
			float4 sampleColor = {255.0f,255.0f,255.0f,0.0f};
			float4 finalColor = {0.0f, 0.0f, 0.0f, 0.0f};
			float4 bgColor = {0.0f, 0.0f, 0.0f, 0.0f};
			float T = 1.0f;
			const float kappa = 0.8f;
			float density, light, dT = 0.f;

			vec = make_float4(closest_wall_dist*dir.x, closest_wall_dist*dir.y, closest_wall_dist*dir.z, 1.f);
			vecAdd(&vec, &camera_pos);

			// TODO: add support for non-uniform resolutions

			scale(&dir, sample_rate);
//			dir.x *= scale_data.x;
//			dir.y *= scale_data.y;
//			dir.z *= scale_data.z;

			while(abs(vec.x) <= (float)box_width/2 + err && abs(vec.y) <= (float)box_height/2 + err && abs(vec.z) <= (float)box_depth/2 + err) {
				float4 tempvec = make_float4(vec.x + (float)box_width/2, vec.y + (float)box_height/2, vec.z + (float)box_depth/2, 1.f);
				density = interpolate(idata, &tempvec, make_int3(box_width, box_height, box_depth), maximum);
				//TODO: rather arbitrary cutoff
				if(density > 0.55f) {
					light = 0.55f; //TODO
					//light = sampleLight(field, vec.x, vec.y, vec.z, dim, ds);
					dT = expf(density * (-sample_rate) * kappa);
					T *= dT;

					float4 temp = scaleColor(sampleColor, (1.0-dT) * T/kappa * light);
					vecAddColor(&finalColor, &temp);
				}

				vecAdd(&vec, &dir);
			}
			finalColor.w = (1.0f-dT);

			finalColor = vecAddColor(scaleColor(finalColor , (1-T)), scaleColor(bgColor, T));

			odata[y*width*4+(x*4)] = 255;
			odata[y*width*4+(x*4)+1] = 255;
			odata[y*width*4+(x*4)+2] = 0;
			odata[y*width*4+(x*4)+3] = 255;

//			odata[y*width*4+(x*4)] = finalColor.x;
//			odata[y*width*4+(x*4)+1] = finalColor.y;
//			odata[y*width*4+(x*4)+2] = finalColor.z;
//			odata[y*width*4+(x*4)+3] = finalColor.w;
		}
		else {
			odata[y*width*4+(x*4)] = 0;
			odata[y*width*4+(x*4)+1] = 0;
			odata[y*width*4+(x*4)+2] = 0;
			odata[y*width*4+(x*4)+3] = 0;
		}
	}
}

string toString(int i) {
	stringstream ss(stringstream::in | stringstream::out);
	ss << i;
	return ss.str();
}

void init() {
	SDL_Init(SDL_INIT_EVERYTHING);
	//surf = SDL_CreateRGBSurface(0, width, height, 32, rmask, gmask, bmask, amask);
	surf = SDL_SetVideoMode(width, height, 32, SDL_HWSURFACE | SDL_DOUBLEBUF);

	block_size.x = 24;
	block_size.y = 18;

	grid_size.x = (width / block_size.x) + 1;
	grid_size.y = (height / block_size.y) + 1;

	scale_data.x = 1.f;
	scale_data.y = 1.f;
	scale_data.z = (float)1/3;

	rotation = make_float3(0, 0, 0);
}

void render() {
	//FIXME: box size with scaling
	volumeRayMarching<<<grid_size, block_size>>>(d, result, height, width, camera_dist, work_dim[0], work_dim[1], work_dim[2], sample_rate, maximum, scale_data, rotation);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete

	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(odata, result, sizeof(char) * width * height * 4, cudaMemcpyDeviceToHost));

	SDL_LockSurface(surf);

	char *p = (char *)surf->pixels;
	memcpy(p, odata, width * height * 4);

	SDL_UnlockSurface(surf);

	SDL_Flip(surf);
}

void OnEvent(SDL_Event* Event) {
	switch(Event->type) {
	case SDL_KEYDOWN:
		switch(keyevent.key.keysym.sym) {
		case SDLK_q:
			SDL_Quit();
			running = false;
			break;
		case SDLK_u:
			render();
			break;
		case SDLK_UP:
			rotation.x += PI/16;
			break;
		case SDLK_DOWN:
			rotation.x -= PI/16;
			break;
		case SDLK_LEFT:
			rotation.y += PI/16;
			break;
		case SDLK_RIGHT:
			rotation.y -= PI/16;
			break;
		case SDLK_PLUS:
			camera_dist -= 25.f;
			break;
		case SDLK_MINUS:
			camera_dist += 25.f;
			break;
		default:
			break;
		}
		break;
	default:
		break;
	}
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 *
 * Note: The parameters in this program are currently set to work with a specific data set
 */
int main(void) {
	unsigned int length;
	unsigned int work_size = work_dim[0]*work_dim[1]*work_dim[2];
	uint16_t *idata;
	char *buffer;
	string fileprefix = "data/CThead.";
	ifstream file;

	idata = new uint16_t[work_size];
	odata = new char[width*height*4];

	init();

	for(int i = 1; i <= work_dim[2]; i++) {
		file.open((fileprefix + toString(i)).c_str(), ifstream::in);

		// get length of file:
		file.seekg (0, ios::end);
		length = file.tellg();
		file.seekg (0, ios::beg);

		buffer = new char[length];

		file.read((char*)buffer, length);

		for(int j = 0; j < work_dim[0]*work_dim[1]; j++) {
			idata[((i-1)*work_dim[0]*work_dim[1])+j] = ((uint16_t*)buffer)[j];
		}

		free(buffer);

		file.close();
	}

	CUDA_CHECK_RETURN(cudaMalloc((void**) &d, sizeof(uint16_t) * work_size));
	CUDA_CHECK_RETURN(cudaMemcpy(d, idata, sizeof(uint16_t) * work_size, cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc((void**) &result, sizeof(int) * width * height * 4));

	bytereverse<<<work_size/512+1, 512>>>(d, work_size);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaMemcpy(idata, d, sizeof(uint16_t) * work_size, cudaMemcpyDeviceToHost));

	for(int i = 0; i < work_size; i++) {
		if(idata[i] > maximum) {
			maximum = idata[i];
		}
	}

	render();

	while(running) {
		while(SDL_PollEvent(&keyevent)) {
			OnEvent(&keyevent);
		}
	}

//	for (int i = 0; i < width * height * 4; i=i+4) {
//		printf("Output: %d %d %d %d\n", (unsigned char)odata[i], (unsigned char)odata[i+1], (unsigned char)odata[i+2], (unsigned char)odata[i+3]);
//	}

	CUDA_CHECK_RETURN(cudaFree((void*) d));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	free(idata);
	free(odata);

	return 0;
}
