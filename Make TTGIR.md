# Triton TTGIR 최적화 단계 분석

`make_ttgir` 함수는 Triton 컴파일 파이프라인에서 TTIR(Triton IR)을 TTGPU IR로 변환하고 최적화하는 핵심 단계.

## 초기 설정 및 준비

1. **레지스터 설정**
   - `maxnreg` 옵션이 제공되면 각 커널에 최대 레지스터 수를 설정
   - 스레드당 사용할 수 있는 32비트 레지스터 수를 제한하는 중요한 성능 튜닝 파라미터

2. **클러스터 정보 설정**
   - NVIDIA의 Hopper 아키텍처부터 지원되는 SM(Streaming Multiprocessor) 클러스터링 기능 활용
   - X, Y, Z 차원의 클러스터 크기 구성으로 멀티 SM 간 통신 최적화

3. **IR에서 TTGPU IR로 변환**
   - `add_convert_to_ttgpuir`: TTIR을 TTGPU IR로 변환하는 핵심 패스
   - 워프 수, 스레드 수, CTA 수 등의 중요한 병렬화 파라미터 지정

## 기본 최적화 패스 적용

모든 GPU 아키텍처에 적용되는 기본 최적화 패스:

1. **메모리 접근 최적화**
   - `add_coalesce`: 스레드 간 메모리 접근 패턴을 코얼레싱하여 대역폭 활용 최대화
   - `add_remove_layout_conversions`: 불필요한 레이아웃 변환 연산 제거

2. **GPU 아키텍처별 특화 최적화**
   - `add_f32_dot_tc`: Ampere(SM80+) 이상에서 FP32 연산의 Tensor Core 활용
   - `add_plan_cta`: NVIDIA 특화 CTA 레이아웃 최적화

3. **병렬 연산 최적화**
   - `add_optimize_thread_locality`: 스레드 간 통신 최소화를 위한 데이터 지역성 향상
   - `add_accelerate_matmul`: 행렬 곱셈 연산 속도 향상을 위한 패턴 최적화
   - `add_optimize_dot_operands`: 행렬 곱셈 연산자 레이아웃 최적화

## 아키텍처별 특화 최적화

코드는 GPU 아키텍처(`capability`)에 따라 다른 최적화 전략을 적용:

### Ampere/Ada 아키텍처 (SM80, SM90)
```python
if capability // 10 in [8, 9]:
    passes.ttgpuir.add_fuse_nested_loops(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_triton_licm(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttgpuir.add_combine_tensor_select_and_if(pm)
    passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
```
- 중첩 루프 융합과 루프 불변 코드 이동(LICM) 최적화
- 조건부 연산(tensor_select 및 if) 결합으로 분기 최소화
- 소프트웨어 파이프라이닝을 통한 메모리 대기 시간 감소

### Hopper 아키텍처 (SM100+)
```python
elif capability // 10 >= 10:
    # 기본 최적화...
    passes.ttgpuir.add_optimize_accumulator_init(pm)
    passes.ttgpuir.add_hoist_tmem_alloc(pm)
    nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)
    passes.ttgpuir.add_warp_specialize(pm, opt.num_stages)
    # 추가 최적화...
```
- 축적기 초기화 최적화로 불필요한 연산 제거
- 텐서 메모리 할당 위치 최적화
- 행렬 곱셈의 왼쪽 피연산자를 텐서 메모리로 승격
- **워프 특화(Warp Specialization)**: 로드/연산 워프 분리로 병렬성 극대화

## 공통 최적화 패스

모든 아키텍처에 다시 적용되는 추가 최적화:

1. **메모리 최적화**
   - `add_prefetch` & `add_WGMMAPrefetch`: 데이터 미리 로드로 메모리 레이턴시 감소
   - `add_coalesce_async_copy`: 비동기 메모리 복사 연산 그룹화

2. **연산 최적화**
   - `add_reduce_data_duplication`: 중복 데이터 처리 제거
   - `add_reorder_instructions`: 명령어 재정렬로 실행 효율성 향상

3. **TMA 최적화 (SM90+)**
   - `add_tma_lowering`: Tensor Memory Accelerator 최적화 
   - `add_fence_insertion`: 메모리 동기화를 위한 펜스 삽입

## 메타데이터 수집 및 반환

1. 클러스터 차원 정보를 메타데이터에 저장
2. 텐서 디스크립터 메타데이터 수집
3. 최적화된 모듈 반환

이 과정을 통해 TTGPU IR은 각 GPU 아키텍처의 특성을 최대한 활용하도록 최적화되며, 다음 단계인 `make_llir`에서 LLVM IR로 변환됨.
