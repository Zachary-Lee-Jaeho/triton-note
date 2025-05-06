# Triton 그리드 정보 컴파일 과정

## 1. Python 단계: 그리드 정의

**관련 파일**: `python/triton/runtime/jit.py`, `python/tutorials/03-matrix-multiplication.py`

### 그리드 지정 방식
- **고정 그리드**:
  ```python
  grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
  kernel[grid](a, b, c, ...)
  ```

- **동적 그리드** (함수를 통한 정의):
  ```python
  grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
  kernel[grid](a, b, c, ...)
  ```

### 그리드 캡처
- `KernelInterface.__getitem__` 메서드가 그리드 정보를 캡처하여 런타임에 사용할 수 있게 함
- `JITFunction.run` 메서드가 그리드 정보를 처리하여 핵심 파라미터 추출

## 2. Triton의 런타임 준비 단계

**관련 파일**: `python/triton/runtime/jit.py`

### 그리드 정규화
```python
# canonicalize grid
assert grid is not None
if callable(grid):
    grid = grid(bound_args)
grid_size = len(grid)
grid_0 = grid[0]
grid_1 = grid[1] if grid_size > 1 else 1
grid_2 = grid[2] if grid_size > 2 else 1
```

### 메타데이터 처리
- 런칭할 때 사용할 메타데이터에 그리드 정보를 포함
```python
launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
```

## 3. 컴파일 단계로 그리드 정보 전달

**관련 파일**: `python/triton/compiler/compiler.py`

### 메타데이터에 저장되는 그리드 관련 정보
```python
class CompiledKernel:
    def __init__(self, src, metadata_group, hash):
        # ...
        self.metadata = KernelMetadata(**metadata)
        # 클러스터 차원 정보 저장
        metadata['cluster_dims'] = tuple(metadata['cluster_dims'])
        # ...
```

- `cluster_dims`: CTA(Cooperative Thread Array) 클러스터 차원 정보
- `num_warps`: 커널당 워프 수
- `num_ctas`: CTA 수

## 4. 백엔드별 그리드 레이아웃 변환

### NVIDIA 백엔드 (CUDA)
**관련 파일**: `third_party/nvidia/backend/compiler.py`

```python
def make_ttgir(mod, metadata, opt, capability):
    # ...
    pm.add_pass("convert-triton-to-triton-gpu{num-warps=" + str(opt.num_warps) + 
                ",num-ctas=" + str(opt.num_ctas) + 
                ",cluster-dims=[" + ",".join([str(d) for d in opt.cluster_dims]) + "]}")
    # ...
```

### AMD 백엔드 (HIP)
**관련 파일**: `third_party/amd/backend/compiler.py`

```python
def make_ttgir(mod, metadata, opt):
    # ...
    pm.add_pass("convert-triton-to-triton-gpu{num-warps=" + str(opt.num_warps) + 
                ",num-ctas=" + str(opt.num_ctas) + "}")
    # ...
```

## 5. Triton IR에서 TritonGPU IR로 변환 (레이아웃 인코딩)

**관련 파일**: `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp`

### `BlockedEncodingAttr` 생성
```cpp
// 그리드 레이아웃 계산과 관련된 파라미터들
triton::gpu::BlockedEncodingAttr::get(
    getContext(), 
    sizePerThread,    // 스레드당 처리할 요소 수
    threadsPerWarp,   // 워프당 스레드 수
    warpsPerCTA,      // CTA당 워프 수
    order,            // 처리 순서
    CTALayoutAttr     // CTA 레이아웃 정보
);
```

### 그리드 정보가 포함된 `CTALayoutAttr` 
```cpp
triton::gpu::CTALayoutAttr::get(
    getContext(), 
    retCTAsPerCGA,   // 그리드 내 CTA 수 (grid_x, grid_y, grid_z)
    retCTASplitNum,  // CTA 분할 번호
    retCTAOrder      // CTA 처리 순서
);
```

## 6. CTA 레이아웃 최적화

**관련 파일**: `lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp`

### 스레드 로컬리티 최적화
```cpp
// 그리드 내에서 CTA 간의 데이터 재사용을 위한 최적화
if (numBlocks == 1) {
  // 단일 블록 케이스 처리
  // ...
} else {
  // 여러 블록에 대한 그룹화 설정
  sizePerThread = {1, N / 2};
  threadsPerWarp = {16, 2};
  warpsPerCTA = {0, 0};
  // M 차원을 따라 최대 워프 그룹 분배
  warpsPerCTA[0] = 4 * std::min(blocksPerTile[0], numWarpGroups);
  // N 차원을 따라 나머지 워프 그룹 분배
  warpsPerCTA[1] = ceil<int>(numWarpGroups, warpsPerCTA[0] / 4);
}
```

## 7. LLVM IR로 변환 단계

**관련 파일**: `lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp`

### 그리드 ID 액세스
```cpp
// 그리드 ID 계산
Value gridIdx[3];
for (int k = 0; k < 3; ++k) {
  gridIdx[k] = rewriter.create<GetProgramIdOp>(loc, k);
}

// 그리드 차원 계산
Value gridDim[2];
for (int k = 0; k < 2; ++k) {
  gridDim[k] = rewriter.create<GetNumProgramsOp>(loc, k);
}

// 선형 인덱스 계산
Value linearId = gridIdx[2];
for (int k = 0; k < 2; ++k) {
  linearId = b.add(gridIdx[1 - k], b.mul(linearId, gridDim[1 - k]));
}
```

## 8. PTX/CUBIN 생성 단계

**관련 파일**: `third_party/nvidia/backend/compiler.py`의 `make_ptx` 및 `make_cubin` 함수

### PTX 코드 생성
```python
def make_ptx(self, src, metadata, opt, capability):
    # ...
    # PTX 코드에 그리드 관련 매개변수 포함
    ptx = llvm.translate_to_ptx(
        mod, 
        triple="nvptx64-nvidia-cuda", 
        features=features, 
        opt_level=opt_level
    )
    # ...
```

### CUBIN 생성
```python
def make_cubin(self, src, metadata, opt, capability):
    # ...
    # CUBIN에 그리드 설정이 내장됨
    cmd = [
        get_ptxas().path, "-v",
        "--gpu-name", sm_arch_from_capability(capability),
        "-o", cubin_path.name,
        "-lineinfo",
        "--output-directory", str(cubin_path.parent.absolute()),
        ptx_path.name
    ]
    # ...
```

## 9. 런타임 단계: 그리드 설정 적용

**관련 파일**: `third_party/nvidia/backend/driver.py`의 `CudaLauncher` 클래스

### 런처 코드 생성
```python
def make_launcher(constants, signature):
    # ...
    src = f"""
    // ...
    static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, 
                        int launch_cooperative_grid, int launch_pdl, 
                        int clusterDimX, int clusterDimY, int clusterDimZ, 
                        int shared_memory, CUstream stream, CUfunction function, 
                        CUdeviceptr global_scratch{', ' + arg_decls if len(arg_decls) > 0 else ''}) {
        void *params[] = {{ {', '.join(params)}
        // ...
    """
```

### 커널 런칭
```python
def __call__(self, gridX, gridY, gridZ, stream, function, *args):
    # ...
    # 최종 디바이스 코드 실행
    self.launch(gridX, gridY, gridZ, stream, function, 
                self.launch_cooperative_grid, self.launch_pdl,
                global_scratch, *args)
```

# 주요 그리드 정보 변환 요약

| 컴파일 단계 | 파일 | 그리드 정보 처리 | 
|------------|------|-----------------|
| 1. 그리드 정의 | `jit.py` | Python 함수/상수로 그리드 크기 정의 | 
| 2. 런타임 준비 | `jit.py` | 그리드 정보 정규화 (grid_0, grid_1, grid_2) |
| 3. 컴파일 메타데이터 | `compiler.py` | 클러스터 차원, 워프 수, CTA 수 메타데이터 저장 |
| 4. 백엔드 변환 | `nvidia/backend/compiler.py` | 백엔드별 그리드 레이아웃 변환 패스 적용 |
| 5. TritonGPU IR 변환 | `TritonToTritonGPUPass.cpp` | 블록 인코딩 및 CTA 레이아웃 속성 생성 |
| 6. 스레드 최적화 | `OptimizeThreadLocality.cpp` | CTA 간 데이터 지역성 최적화 |
| 7. LLVM IR 변환 | `ConvertLayoutOpToLLVM.cpp` | 그리드 ID 및 차원 접근 코드 생성 |
| 8. PTX/CUBIN 생성 | `nvidia/backend/compiler.py` | 그리드 정보가 포함된 디바이스 코드 생성 |
| 9. 런타임 실행 | `nvidia/backend/driver.py` | 최종 그리드 설정으로 커널 런칭 |
