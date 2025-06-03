// hard_voxelize.cu
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
namespace py = pybind11;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T>
__global__ void dynamic_voxelize_kernel(
    const T* points, int* coors, 
    const float voxel_x, const float voxel_y, const float voxel_z, 
    const float coors_x_min, const float coors_y_min, const float coors_z_min, 
    const float coors_x_max, const float coors_y_max, const float coors_z_max, 
    const int grid_x, const int grid_y, const int grid_z, 
    const int num_points, const int num_features, const int NDim) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {

    auto points_offset = points + index * num_features;
    auto coors_offset = coors + index * NDim;

    int c_x = floor((points_offset[0] - coors_x_min) / voxel_x);
    int c_y = floor((points_offset[1] - coors_y_min) / voxel_y);
    int c_z = floor((points_offset[2] - coors_z_min) / voxel_z);

    if (c_x < 0 || c_x >= grid_x) {
      coors_offset[0] = -1;
    }
    else if (c_y < 0 || c_y >= grid_y) {
      coors_offset[0] = -1;
    }
    else if (c_z < 0 || c_z >= grid_z) {
      coors_offset[0] = -1;
    } 
    else {
      coors_offset[0] = c_x;
      coors_offset[1] = c_y;
      coors_offset[2] = c_z;
    }
  }
}

// MurmurHash3 32-bit finalizer style
static __device__ uint64_t hash_func(uint64_t k) {
  k ^= k >> 16;
  k *= 0x85ebca6bL;
  k ^= k >> 13;
  k *= 0xc2b2ae35L;
  k ^= k >> 16;
  return k;
}

static __device__ void insert_hash(int* hash_table, int* count, const int hash_size, const int key) {
  uint64_t hash_value = hash_func((uint64_t)key);
  uint64_t half = hash_size / 2;
  int slot = int(hash_value % half);
  while (true) {
    // atomicCAS(addr, compare, val):
    //   *addr == compare 이면 *addr = val; return (이전 *addr)
    int pre_key = atomicCAS(hash_table + slot, -1, key);
    if (pre_key == -1) {
      // 빈 슬롯에 성공적으로 key 삽입 → 이 key의 ID를 뒷영역(slot+half)에 저장
      // atomicAdd은 old value(=새 ID)를 반환
      hash_table[slot + half] = atomicAdd(count, 1);
      break;
    } 
    else if (pre_key == key) {
      // 이미 같은 key가 있으면 중복 삽입 건너뛰기
      break;
    }
    else{
      // 다른 key가 있으면 슬롯을 바꿔서 재시도
      slot = (slot + 1) % half;
    }
  }
}

static __device__ int lookup_hash(const int key, const int hash_size, const int* hash_table){
  uint64_t hash_value = hash_func((uint64_t)key);
  uint64_t half = hash_size / 2;
  int slot = int(hash_value % half);

  while (true) {
    int pre_key = hash_table[slot];
    if (pre_key == key) {
      return hash_table[slot + half];
    } 
    else if (pre_key == -1) {
      return -1;
    }
    else{
      slot = (slot + 1) % half;
    }
  }
}

__global__ void build_hash_table(
    const int* coor, int* hash_table, int* voxel_num, const int hash_size,
    const int grid_x, const int grid_y, const int grid_z, const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {

    auto coor_offset = coor + index * NDim;
    if (coor_offset[0] == -1) return;
    int hash_idx = coor_offset[2] * grid_y * grid_x + coor_offset[1] * grid_x + coor_offset[0];
    insert_hash(hash_table, voxel_num, hash_size, hash_idx);
  }
}

__global__ void build_point_table(
    const int* coor, const int* hash_table, int* num_points_per_voxel, int* point_table, 
    const int hash_size, const int grid_x, const int grid_y, const int grid_z,
    const int max_points, const int voxel_num, const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {

    // 각 포인트에 대해 voxel 좌표 가져오기
    auto coor_offset = coor + index * NDim;
    if (coor_offset[0] == -1) return;

    // lookup_hash_table에서 voxelidx 가져오기
    int hash_idx = coor_offset[2] * grid_y * grid_x + coor_offset[1] * grid_x + coor_offset[0];
    int voxel_id = lookup_hash(hash_idx, hash_size, hash_table);
    if (voxel_id == -1 || voxel_id >= voxel_num) return;

    // AtomicAdd를 통해 voxel 좌표에 해당하는 포인트 ID를 지정
    // max_point 이하일 경우 point_table에 포인트 ID 저장
    // max_point 초과할 경우 skip
    int point_idx = atomicAdd(num_points_per_voxel + voxel_id, 1);
    if (point_idx < max_points) {
      point_table[voxel_id * max_points + point_idx] = index;
    }
  }
}

template <typename T>
__global__ void copy_voxel_feature(
    const T* points, const int* point_table, const int* temp_coor, T* voxels, int* coors,
    const int* num_points_per_voxel_temp, int* num_points_per_voxel,
    const int max_points, const int num_features, const int voxel_num, const int NDim) {
  CUDA_1D_KERNEL_LOOP(index, voxel_num * max_points) {

    int voxel_idx = index / max_points;
    int voxel_point_idx = index % max_points;
    int point_index = point_table[index];
    if (voxel_idx >= voxel_num) return;
    if (point_index == -1) return;

    if (voxel_point_idx == 0){
      auto coors_offset = coors + voxel_idx * NDim;
      auto temp_coor_offset = temp_coor + point_index * NDim;

      // voxel 좌표 복사
      coors_offset[0] = temp_coor_offset[0];
      coors_offset[1] = temp_coor_offset[1];
      coors_offset[2] = temp_coor_offset[2];

      // num_points_per_voxel 복사
      num_points_per_voxel[voxel_idx] = min(max_points, num_points_per_voxel_temp[voxel_idx]);
    }

    // point feature 복사
    for (int i = 0; i < num_features; ++i) {
      auto voxels_offset = voxels + index * num_features;
      auto points_offset = points + point_index * num_features;
      voxels_offset[i] = points_offset[i];
    }
  }
}

namespace voxelization {

int hard_voxelize_gpu2(const at::Tensor& points, at::Tensor& voxels,
  at::Tensor& coors, at::Tensor& num_points_per_voxel,
  const std::vector<float> voxel_size,
  const std::vector<float> coors_range,
  const int max_points, const int max_voxels,
  const int NDim = 3) {
    CHECK_INPUT(points);

    at::cuda::CUDAGuard device_guard(points.device());
  
    const int num_points = points.size(0);
    const int num_features = points.size(1);
  
    const float voxel_x = voxel_size[0];
    const float voxel_y = voxel_size[1];
    const float voxel_z = voxel_size[2];
    const float coors_x_min = coors_range[0];
    const float coors_y_min = coors_range[1];
    const float coors_z_min = coors_range[2];
    const float coors_x_max = coors_range[3];
    const float coors_y_max = coors_range[4];
    const float coors_z_max = coors_range[5];
  
    const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
    const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
    const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

    // 1. 각 포인트에 대해 voxel 좌표 계산
    auto temp_coors = at::zeros({num_points, NDim}, points.options().dtype(at::kInt));
  
    dim3 grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
    dim3 block(512);

    AT_DISPATCH_ALL_TYPES(
        points.scalar_type(), "hard_voxelize_kernel", ([&] {
          dynamic_voxelize_kernel<scalar_t>
              <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                  points.contiguous().data_ptr<scalar_t>(),
                  temp_coors.contiguous().data_ptr<int>(), 
                  voxel_x, voxel_y, voxel_z, 
                  coors_x_min, coors_y_min, coors_z_min, 
                  coors_x_max, coors_y_max, coors_z_max, 
                  grid_x, grid_y, grid_z, num_points, num_features, NDim);
        }));
    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());
  
    // 2. voxel 좌표 -> voxelidx 해시 테이블 생성
    int total_max_points = 300000; // 모든 frame에서 voxel의 최대 개수가 300000개 이하임을 확인
    auto hash_tables = -at::ones({total_max_points * 4}, points.options().dtype(at::kInt));
    auto voxel_num = at::zeros({1}, points.options().dtype(at::kInt));
    int hash_size = num_points * 4;

    dim3 hash_grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
    dim3 hash_block(512);

    build_hash_table
      <<<hash_grid, hash_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        temp_coors.contiguous().data_ptr<int>(),
        hash_tables.contiguous().data_ptr<int>(),
        voxel_num.contiguous().data_ptr<int>(),
        hash_size, grid_x, grid_y, grid_z, num_points, NDim);
    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());

    // 3. voxelidx마다 point 배열 생성
    auto voxel_num_cpu = voxel_num.to(at::kCPU);
    int voxel_num_int = voxel_num_cpu.data_ptr<int>()[0];
    voxel_num_int = min(voxel_num_int, max_voxels);
    auto point_tables = -at::ones({voxel_num_int * max_points}, points.options().dtype(at::kInt));
    auto num_points_per_voxel_temp = at::zeros({voxel_num_int}, points.options().dtype(at::kInt));

    dim3 map_grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
    dim3 map_block(512);

    build_point_table
      <<<map_grid, map_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        temp_coors.contiguous().data_ptr<int>(),
        hash_tables.contiguous().data_ptr<int>(),
        num_points_per_voxel_temp.contiguous().data_ptr<int>(),
        point_tables.contiguous().data_ptr<int>(),
        hash_size, grid_x, grid_y, grid_z, max_points, voxel_num_int, num_points, NDim);
    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());

    // 4. point feature -> voxel feature/좌표 복사
    dim3 cp_grid(std::min(at::cuda::ATenCeilDiv(voxel_num_int * max_points, 512), 4096));
    dim3 cp_block(512);
    AT_DISPATCH_ALL_TYPES(
        points.scalar_type(), "copy_voxel_feature", ([&] {
          copy_voxel_feature<scalar_t>
            <<<cp_grid, cp_block, 0, at::cuda::getCurrentCUDAStream()>>>(
              points.contiguous().data_ptr<scalar_t>(),
              point_tables.contiguous().data_ptr<int>(),
              temp_coors.contiguous().data_ptr<int>(),
              voxels.contiguous().data_ptr<scalar_t>(),
              coors.contiguous().data_ptr<int>(),
              num_points_per_voxel_temp.contiguous().data_ptr<int>(),
              num_points_per_voxel.contiguous().data_ptr<int>(),
              max_points, num_features, voxel_num_int, NDim);
        }));
    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());

    return voxel_num_int;
}

}