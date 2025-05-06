# make_ttir 단계의 패스들과 역할

`make_ttir` 함수는 Triton IR(TTIR)을 최적화하는 단계로, Python AST에서 생성된 초기 TTIR을 더 효율적인 형태로 변환함. 

## 1. add_inliner

**역할**: 함수 인라인화
- 함수 호출 오버헤드 제거
- 인라인된 코드를 통해 추가적인 최적화 기회 제공

**예시**:
```python
@triton.jit
def add_one(x):
    return x + 1

@triton.jit
def main(X, Y, n_elements):
    pid = tl.program_id(0)
    i = pid * 128 + tl.arange(0, 128)
    x = tl.load(X + i)
    y = add_one(x)  # 함수 호출
    tl.store(Y + i, y)
```
→ 인라인화 후:
```python
@triton.jit
def main(X, Y, n_elements):
    pid = tl.program_id(0)
    i = pid * 128 + tl.arange(0, 128)
    x = tl.load(X + i)
    y = x + 1  # 인라인됨
    tl.store(Y + i, y)
```

## 2. add_rewrite_tensor_pointer

**역할**: 텐서 포인터 연산 최적화
- 텐서 포인터 관련 연산을 더 효율적인 형태로 변환
- 복잡한 인덱싱 패턴 단순화

**예시 IR**:
```mlir
%ptr = tt.make_tensor_ptr %base, [%shape0, %shape1], [%stride0, %stride1], [%offset0, %offset1]
```
→ 최적화된 형태로 변환, 불필요한 인덱싱 계산 제거

## 3. add_canonicalizer

**역할**: 표준 IR 정규화
- 일반적인 패턴 단순화 (상수 폴딩, 중복 연산 제거 등)
- 표준 형태로 연산 변환

**예시 IR**:
```mlir
%0 = arith.addi %a, %a
%1 = arith.muli %0, %b
```
→ 정규화 후:
```mlir
%0 = arith.muli %a, %c2
%1 = arith.muli %0, %b
```

## 4. add_combine

**역할**: 작은 연산들을 더 큰 연산으로 결합
- 여러 산술 연산을 단일 복합 연산으로 결합
- 메모리 접근 패턴 최적화

**예시**:
```mlir
%0 = tt.load %ptr
%1 = arith.addf %0, %cst1
%2 = arith.mulf %1, %cst2
```
→ 결합 후:
```mlir
%0 = tt.load %ptr
%2 = tt.fused_add_mul %0, %cst1, %cst2
```

## 5. add_reorder_broadcast

**역할**: 브로드캐스트 연산 순서 최적화
- 브로드캐스트와 다른 연산의 순서 재배치
- 계산 효율성 향상을 위한 순서 최적화

**예시 IR**:
```mlir
%0 = tt.broadcast %x, 0
%1 = arith.addf %0, %y
```
→ 최적화 후 (가능한 경우):
```mlir
%0 = arith.addf %x, %y_reduced
%1 = tt.broadcast %0, 0
```

## 6. add_cse

**역할**: 공통 부분식 제거(Common Subexpression Elimination)
- 중복 계산 제거
- 레지스터 사용량 최적화

**예시 IR**:
```mlir
%0 = arith.mulf %a, %b
%1 = arith.addf %c, %d
%2 = arith.mulf %a, %b  // 중복 계산
%3 = arith.addf %1, %2
```
→ 최적화 후:
```mlir
%0 = arith.mulf %a, %b
%1 = arith.addf %c, %d
%3 = arith.addf %1, %0  // %0 재사용
```

## 7. add_symbol_dce

**역할**: 사용되지 않는 심볼 제거(Dead Code Elimination)
- 미사용 변수, 함수, 연산 제거
- 코드 크기 감소 및 성능 향상

**예시 IR**:
```mlir
%0 = arith.addf %a, %b
%1 = arith.mulf %c, %d  // 사용되지 않음
%2 = arith.subf %a, %e
tt.return %0, %2
```
→ 최적화 후:
```mlir
%0 = arith.addf %a, %b
%2 = arith.subf %a, %e
tt.return %0, %2
```

## 8. add_loop_unroll

**역할**: 루프 펼치기
- 반복문 오버헤드 감소
- 명령어 수준 병렬성 증가

**예시 코드**:
```python
@triton.jit
def kernel(X, Y):
    for i in range(4):
        Y[i] = X[i] * 2
```
→ 펼친 후:
```python
@triton.jit
def kernel(X, Y):
    Y[0] = X[0] * 2
    Y[1] = X[1] * 2
    Y[2] = X[2] * 2
    Y[3] = X[3] * 2
```

## 전체 `make_ttir` 함수 정의

`third_party/nvidia/backend/compiler.py`에 구현된 `make_ttir` 함수:

```python
@staticmethod
def make_ttir(mod, metadata, opt):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_pointer(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(mod)
    return mod
```

이 함수는 MLIR 패스 매니저를 설정하고 위에서 설명한 최적화 패스들을 순서대로 적용한 후, 최적화된 모듈을 반환. 이러한 최적화 패스들을 통해 Triton IR이 더 효율적인 형태로 변환되어 다음 단계인 TritonGPU IR 변환으로 넘어감.
