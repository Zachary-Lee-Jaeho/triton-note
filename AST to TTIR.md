# ast_to_ttir 과정에서 정의된 패스와 역할

Triton의 Python DSL에서 AST(Abstract Syntax Tree)가 TTIR(Triton IR)로 변환되는 과정에서 다양한 컴파일러 패스들이 사용됨
## 1. Python에서 TTIR로 변환 과정

`ast_to_ttir` 함수는 Python AST를 TTIR로 변환하는 핵심 단계. 이 함수는 `python/triton/compiler/code_generator.py`에 정의되어 있음:

1. **AST 함수 분석**: 함수 인자와 타입 정보 추출
2. **CodeGenerator 생성**: Python AST를 순회하여 TTIR로 변환하는 방문자(visitor) 패턴 구현
3. **AST 변환**: Python AST를 방문하며 TTIR 생성

## 2. TTIR 최적화 패스

TTIR 생성 후 `make_ttir` 함수에서 다음과 같은 패스들이 적용:

### 2.1 `add_inliner`
- **역할**: 함수 호출을 인라인화하여 함수 호출 오버헤드 제거
- **정의**: MLIR 표준 인라이너 패스를 사용

### 2.2 `add_rewrite_tensor_pointer`
- **역할**: 텐서 포인터 관련 연산을 더 효율적인 형태로 재작성
- **정의**: `passes.ttir.add_rewrite_tensor_pointer`

### 2.3 `add_canonicalizer`
- **역할**: 표준 정규화 패스로, 불필요한 연산 제거 및 연산 단순화
- **정의**: MLIR 표준 패스

### 2.4 `add_combine`
- **역할**: 여러 기본 연산을 더 효율적인, 더 큰 연산으로 결합
- **정의**: `passes.ttir.add_combine`

### 2.5 `add_reorder_broadcast`
- **역할**: 브로드캐스트 연산 순서 최적화
- **정의**: `passes.ttir.add_reorder_broadcast`

### 2.6 `add_cse`
- **역할**: Common Subexpression Elimination, 중복 연산 제거
- **정의**: MLIR 표준 패스

### 2.7 `add_symbol_dce`
- **역할**: 사용되지 않는 심볼 제거 (Dead Code Elimination)
- **정의**: MLIR 표준 패스

### 2.8 `add_loop_unroll`
- **역할**: 반복문을 풀어서 루프 오버헤드 감소
- **정의**: `passes.ttir.add_loop_unroll`

## 3. TritonIR에서 TritonGPU IR로 변환

`add_convert_to_ttgpuir` 패스는 TTIR을 TTGIR(Triton GPU IR)로 변환하는 핵심 패스:

- **역할**: Triton IR의 디바이스 독립적 표현을 GPU에 최적화된 표현으로 변환
- **파라미터**:
  - `cuda:{capability}`: 타겟 GPU 아키텍처
  - `num_warps`: 커널에서 사용할 워프 개수
  - `32`: 워프당 스레드 수 (NVIDIA GPU에서 고정)
  - `num_ctas`: CTA(Cooperative Thread Array) 개수

이 패스는 다음과 같은 주요 변환을 수행:
- 메모리 계층 구조 매핑 (전역/공유/레지스터 메모리)
- 스레드 블록 및 워프 레벨 병렬화
- 도트 프로덕트와 같은 특수 연산의 하드웨어 가속 매핑

## 4. TTIR에서 TTGIR을 거쳐 디바이스 코드로의 변환 흐름

전체 변환 흐름을 정리하면:

1. **Python AST → TTIR**: `ast_to_ttir` 함수가 수행
2. **TTIR 최적화**: `make_ttir`에서 다양한 최적화 패스 적용
3. **TTIR → TTGIR**: `add_convert_to_ttgpuir` 패스가 수행
4. **TTGIR 최적화**: `make_ttgir`에서 디바이스 특화 최적화 패스 적용
5. **TTGIR → LLVM IR**: `make_llir`에서 디바이스별 코드 생성 준비
6. **LLVM IR → PTX/AMDGCN**: 디바이스별 어셈블리 생성
7. **PTX/AMDGCN → 디바이스 바이너리**: `make_cubin`/`make_hsaco`에서 최종 바이너리 생성

이 전체 과정을 통해 Triton의 Python 코드가 고성능 GPU 바이너리로 변환됨.
