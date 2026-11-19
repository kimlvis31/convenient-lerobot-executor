# convLexecutor

[English README](README.md)

LeRobot의 학습·평가·재개 작업을 실행하는 도구입니다. 대화형 메뉴로 작업을 설정하거나 파일에서 작업 큐를 가져올 수 있습니다. 예제에는 기본 모델 평가, 저장된 체크포인트 평가, 초기화 학습, 기존 가중치에서 새 최적화 상태로 이어 학습, 중단 지점에서 정확히 재개하는 작업이 들어 있습니다.

**AI 작성 고지.** 실행 관리자, 예제 큐, 준비 스크립트, 이 프로젝트의 TinyVLA(Vision-Language-Action, 시각·언어·행동 모델) 연결 코드 등 프로젝트 자체의 실행 스크립트는 모두 AI(Artificial Intelligence, 인공지능)가 작성했습니다. 함께 제공하는 상위 프로젝트 코드는 원래 작성자와 고지를 유지합니다. [외부 소스와 라이선스](THIRD_PARTY.md)

## 직접 설치하기

아래 명령은 Git과 Conda를 설치한 Linux의 Bash 터미널에서 실행합니다.

실행 관리자는 `convLexecutor/`의 `scripts/`에서 작업 큐를, `datasets/`에서 데이터를, `externals/`에서 모델 연결을 읽습니다. 결과는 `outputs/`에 저장합니다. LeRobot은 선택한 Conda 환경에서 실행합니다. 기존 소스 폴더에 연결된 설치도 사용할 수 있습니다.

검증 환경은 Linux, Conda, LeRobot 0.6.2의 `145cc7ea53e5864bd5efec4e9bb70ee799e47bd8` 소스 버전, bfloat16을 지원하는 NVIDIA 장치입니다. 기존 호환 환경이 있으면 재사용합니다. 새 환경은 아래 명령으로 만들 수 있습니다. LIBERO(Lifelong Learning Benchmark on Robot Manipulation, 로봇 조작 평생학습 벤치마크)의 환경 요구 사항은 공식 문서에서 확인할 수 있습니다. [영문 설치 안내](README.md#manual-setup) · [LIBERO 설치](https://huggingface.co/docs/lerobot/libero)

TinyVLA(Vision-Language-Action, 시각·언어·행동 모델)는 별도의 Conda 환경을 사용합니다. 모델 환경 이름은 `tinyvla_infer` 또는 `lerobot_tinyvla`입니다. `lerobot` 환경에는 이 저장소의 평가 연결 패키지를 설치합니다.

```bash
git clone https://github.com/kimlvis31/convenient-lerobot-executor.git ~/convLexecutor
cd ~/convLexecutor
conda create -n lerobot python=3.12 pip
conda run -n lerobot python -m pip install \
  "lerobot[libero] @ git+https://github.com/huggingface/lerobot.git@145cc7ea53e5864bd5efec4e9bb70ee799e47bd8"
conda run -n lerobot lerobot-train --help

conda create -n tinyvla_infer python=3.10 pip
conda run -n tinyvla_infer python -m pip install -r externals/TinyVLA/sources/TinyVLA/requirements.txt
conda run -n tinyvla_infer python -c "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()"
conda run -n lerobot python -m pip install --no-deps --no-build-isolation --editable \
  "./externals/TinyVLA/lerobot_policy_tinyvla"
```

평가 연결 패키지를 편집 가능 모드로 설치하면 `git pull`로 응용 프로그램과 연결 코드가 함께 갱신되며, 이전에 패키지 복사본으로 설치했다면 위 연결 패키지 설치 명령을 한 번 다시 실행하면 됩니다.

LeRobot 기본 모델만 사용한다면 별도 모델 환경과 아래 TinyVLA(Vision-Language-Action, 시각·언어·행동 모델) 예제 준비 단계는 건너뛸 수 있습니다.

## 실습 자산 준비와 실행

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot
python -B ~/convLexecutor/executor/prepare_examples.py --dry-run
python -B ~/convLexecutor/executor/prepare_examples.py
python -B ~/convLexecutor/executor/manage.py --inventory
python -B ~/convLexecutor/executor/manage.py \
  --queue ~/convLexecutor/scripts/example_script.py
```

준비 스크립트는 버전을 고정한 공개 기반 가중치와 크림치즈 집기 시연 1개를 사용하는 데 필요한 파일을 받습니다. 행동 출력부를 4단계 학습해 `outputs/example/`에 4단계 모델과 2단계 재개 상태를 만듭니다. 재개 상태에는 해당 컴퓨터의 기반 모델 위치가 들어가므로 각 컴퓨터에서 준비 명령을 실행합니다.

직접 설정하려면 `manage.py`를 실행하고 **Configure a new job**을 선택합니다. 한 목록에서 모델을 고르고 평가 벤치마크나 LIBERO(Lifelong Learning Benchmark on Robot Manipulation, 로봇 조작 평생학습 벤치마크) 스위트를 선택합니다. Enter를 누르면 표시된 기본값이 적용됩니다. 준비된 작업 큐를 실행하려면 **Import a job queue script**를 선택합니다. [예제 설정](scripts/example_script.py)

작업을 시작하면 convLexecutor가 선택한 모델과 벤치마크를 함께 실행할 수 있는 Conda 환경을 찾아 재사용합니다. π 계열의 SentencePiece처럼 현재 LeRobot 버전이 선언한 패키지가 빠져 있으면 필요한 것만 설치합니다. 기존 버전을 유지한 상태에서 의존성이 충돌하면 별도 환경을 만듭니다. 관리자가 설치 명세를 제공하는 소스형 벤치마크는 실행을 확정한 작업에서 처음 필요할 때 자동으로 준비합니다. 미리보기에서는 설치하지 않습니다. 여전히 **Manual Setup**이 필요한 벤치마크에는 부족한 실행 조건을 표시하며, 설치된 LeRobot 환경 플러그인은 자동으로 조회합니다. 체크포인트는 작업 설정에서 선택합니다. PushT와 ALOHA는 작업 하나당 태스크 하나를 평가합니다. 배치 추론은 유지하고 시뮬레이터 단계는 기본적으로 동기 실행합니다. Meta-World는 시뮬레이터의 기본 에피소드 길이를 사용합니다.

의존성 충돌로 관리자가 새로 만든 복제 환경은 원본의 `conda-meta/state.env_vars` 선언을 자식 프로세스 메모리에서 이어받습니다. 지원하는 모델 접근·캐시 설정을 유지하면서 대상 환경의 제어 설정과 프로젝트 데이터셋 위치를 보존합니다. 복제 환경에는 원본 환경 위치만 기록하며 인증값을 그 기록·실행 설정·로그로 복사하지 않습니다. 선언된 변수의 상속을 지원하며 원본 `activate.d`의 임의 스크립트나 그 스크립트가 계산하는 설정을 다시 실행하지는 않습니다.

지원되는 Linux 구성에서 FLUX.3 실행 준비를 확정하면 선택한 Torch 설치를 바꾸지 않고 호환되는 공식 `natten` 바이너리를 준비해 불러오기를 확인하며, 지원하지 않는 조합은 소스를 임의로 컴파일하는 대신 필요한 조건을 안내합니다. [공식 확장 라이브러리 설치 조건](https://natten.org/install/)

정책이나 그 모델이 사용하는 하위 모델에 승인이 필요하면 Settings에서 알려진 저장소 링크를 볼 수 있습니다. 예를 들어 GR00T N1.7은 정책 체크포인트가 공개 상태여도 Cosmos-Reason2-2B 접근 권한을 요구할 수 있습니다. 실행 컴퓨터에서 쓰는 계정으로 안내된 저장소를 열어 약관에 동의하거나 권한을 신청하세요. 승인 후 해당 저장소를 읽을 수 있는 토큰으로 다시 실행하면 됩니다. 실행 도중 새로 발견된 저장소의 권한 오류는 Result에 표시됩니다. [Hugging Face 모델 접근 안내](https://huggingface.co/docs/hub/models-gated)

평가 동작 가능 여부는 패키지 설치 여부와 별도로 검사합니다. 확인한 기본 Unitree G1 MuJoCo 모듈은 외부 DDS(Data Distribution Service, 데이터 분산 서비스)로 모터 명령을 받으며, `step(action)`이 정책 행동을 적용하거나 태스크 성공·에피소드 종료를 정의하지 않습니다. 관리자는 기본 모듈의 기본 선택, `main`, 확인한 버전을 의존성 준비 전에 중단하고 태스크 어댑터가 필요함을 설명합니다. `--env.hub_path=owner/task-environment` 또는 LeRobot 환경 플러그인으로 실제 태스크 환경을 지정할 수 있습니다. 그 밖의 명시한 버전은 평가 능력 미확인 안내와 함께 선택할 수 있으며, 행동 적용·성공·종료 동작은 별도로 검증해야 합니다. [확인한 Unitree 소스](https://huggingface.co/lerobot/unitree-g1-mujoco/blob/b00f8b114c3d5d98b44b7ad0285d631af73f4bc5/env.py)

공개된 `lerobot/FineART-VLA-subtask-midtrained` 체크포인트와 기본 양팔 환경(`aloha`)을 함께 선택하면 Settings에 확인된 관절·제어 차이를 안내합니다. 체크포인트는 팔마다 팔 관절 5개와 그리퍼 값 2개를, 기본 양팔 환경은 팔 관절 6개와 정규화한 그리퍼 값 1개를 사용합니다. 양쪽 모두 14개 값이어도 관절 순서·단위·제어 의미가 같다는 뜻은 아니며 카메라 이름 변경만으로 해결되지 않습니다. 해당 환경에 맞게 추가 학습한 체크포인트나 의미가 일치하는 태스크 어댑터를 사용해야 합니다. 이 안내는 확인한 공개 체크포인트와 기본 환경의 조합에 해당하며, 별도 학습한 체크포인트나 사용자 환경을 일괄 금지하지 않습니다. [공개 FineART 체크포인트](https://huggingface.co/lerobot/FineART-VLA-subtask-midtrained)

로컬 학습 데이터셋은 폴더를 선택하면 됩니다. `selection.json`에 저장소 이름과 학습 에피소드 목록이 있으면 자동으로 사용합니다. TinyVLA는 로컬 학습 분할의 태스크가 선택한 태스크와 일치하면 해당 분할을 자동으로 사용합니다. **Use a dataset repository**를 고르면 Hugging Face 저장소 이름을 입력합니다.

정책 목록은 로컬 체크포인트가 없는 모델을 흐리게 표시합니다. Hugging Face 캐시도 로컬에 포함합니다. 평가에는 선택한 벤치마크에 맞는 체크포인트를 사용합니다. LeRobot 모델은 현재 버전의 정책 설정이 지원하는 학습 범위와 LoRA(Low-Rank Adaptation, 저랭크 적응), 학습 옵션을 보여줍니다. TinyVLA는 행동 헤드만 학습하거나 행동 헤드와 언어 어텐션에 LoRA(Low-Rank Adaptation, 저랭크 적응)를 함께 적용할 수 있습니다. 정확한 재개는 기존 설정과 옵티마이저 상태를 복원합니다.

검증된 체크포인트의 저장된 카메라 별칭과 알려진 가상 카메라 슬롯은 원래 설정과 마스크를 보존하면서 자동으로 연결하며, 사용자가 명시한 카메라·정책 설정을 우선합니다.

Training Scope는 선택한 모델의 학습 범위 조합을 한 목록에 표시합니다. 학습 전에 로컬 데이터셋과 체크포인트의 입력을 확인하고, 자동으로 추가한 필수 인자는 Settings에서 초록색으로 표시합니다. 데이터셋 원본은 보존하며 실행 실패 시 Result에서 실패 원인과 전체 로그 경로를 볼 수 있습니다.

각 작업의 실제 실행 파일 `run.py`, 다시 가져올 수 있는 `script.py`, 시작·종료·소요 시간과 단계별 콘솔 로그는 각 결과 폴더의 `executor/`에 기록합니다. 평가 결과는 `outputs/evaluation/`, 학습 결과는 `outputs/training/`에 저장합니다. Git으로 관리하는 파일은 소스, 설정과 출처 목록입니다. 데이터, 가중치, 실행 로그와 결과는 로컬에서 생성합니다.

모델을 선택할 때는 번호를 직접 입력합니다. [모델별 포맷](scripts/)에는 학습 작업과 평가 작업이 각각 하나씩 들어 있으며, 필수·선택 항목과 모델별 선행 조건을 주석으로 표시합니다.

**Save Job Outputs?**에서 **N**을 선택하면 임시 폴더에서 실행하고 결과는 터미널에 표시합니다. 파이썬 작업 큐에서는 개별 작업에 `"save_outputs": False`를 지정합니다. 실행이 끝나면 임시 결과와 로그를 정리합니다. 학습 후 평가에서는 평가가 끝날 때까지 체크포인트를 임시로 사용합니다.

**Settings**에서 각 단계의 Conda 환경, 선택 이유, 모델과 시뮬레이터의 요구 패키지와 확인된 버전을 볼 수 있습니다. TinyVLA는 모델 실행 환경과 LeRobot 데이터 읽기 또는 평가 환경을 각각 표시합니다. `unverified`는 확인하지 못한 항목입니다. `--dry-run`으로 설치 없이 미리 볼 수 있습니다. 실행 준비 중 환경이 달라지면 최종 선택을 표시하고 실행 스크립트에 기록합니다.

설치 작업을 에이전트에게 맡길 때는 [AGENTS.md](AGENTS.md)를 사용하세요.

## RoboMME 평가

대화형 평가에서 **RoboMME**를 선택하거나 네이티브 작업의 `eval.env`를 `"robomme"`로 지정한다. `eval.robomme_runtime`의 기본값은 `"auto"`다. 실행기는 정책과 시뮬레이터를 함께 실행할 수 있는 환경을 먼저 찾는다. 의존성이 충돌하면 기존 LeRobot 환경에서 정책을 실행하고 별도 환경에서 시뮬레이터를 실행한다. 영상, 통계, 실행 로그는 공식 LeRobot 평가기가 생성한다.

처음 분리 실행할 때 기존 Conda 환경을 검색하고, 필요한 경우 Linux 사용자 캐시에 시뮬레이터 환경을 새로 만든다. 자동 설치에는 인터넷 연결, CUDA(Compute Unified Device Architecture, 엔비디아 병렬 연산 플랫폼) 12.6 이상을 지원하는 드라이버, 작동하는 Vulkan 렌더러가 필요하다. 실제 물리 한 스텝을 확인한 환경을 재사용한다. 저장 위치는 `CONVLEX_ROBOMME_CACHE`로 지정하고 실행 화면에서 시뮬레이터 Python 경로를 확인할 수 있다.

합쳐진 환경을 명시적으로 요구하려면 `eval.robomme_runtime`을 `"shared"`로, 두 프로세스를 사용하려면 `"split"`으로 지정한다. `eval.simulator_environment`에는 기존 시뮬레이터 Conda 환경 이름이나 절대 경로를, `eval.environment`에는 정책 환경을 지정한다. [공식 RoboMME 설치 및 태스크 번호 안내](https://huggingface.co/docs/lerobot/main/robomme)

## 라이선스

프로젝트 자체 코드는 MIT(Massachusetts Institute of Technology, 매사추세츠 공과대학) 라이선스를 따릅니다. 외부 소스와 공개 데이터는 각각의 원래 라이선스를 따릅니다. [프로젝트 라이선스](LICENSE) · [외부 소스와 라이선스](THIRD_PARTY.md)

## RoboTwin 실행 환경

Linux x86-64에서 RoboTwin 평가 실행을 확정하면 재사용할 수 있는 검증된 환경을 먼저 찾고, 없으면 고정된 소스 버전, 로봇·물체 자산, 호환 시뮬레이터 환경을 자동으로 준비한다. 시뮬레이터는 별도 파이썬 프로세스에서 실행하므로 NumPy와 네이티브 라이브러리가 정책 환경의 패키지를 대체하지 않는다. 준비 항목에는 호환되는 CUDA(Compute Unified Device Architecture, NVIDIA 병렬 연산 플랫폼)용 PyTorch, SAPIEN, MPlib, PyTorch3D, CuRobo와 필요한 상위 소스 호환 수정이 포함된다. NVIDIA 드라이버와 작동하는 Vulkan 렌더러가 필요하다. 실제 카메라 렌더링과 물리 계산 한 단계를 통과한 환경만 사용 가능 상태로 표시한다. 정책 평가와 체크포인트의 카메라 연결은 설치된 LeRobot 평가기를 사용한다.

기존 환경은 `ROBOTWIN_ROOT` 또는 검색 가능한 `envs` 패키지로 계속 사용할 수 있다. 실행기는 필수 파일과 MPlib·NumPy 호환성을 검사하고 시뮬레이터 자산 루트에서 실행한다. [고정된 상위 통합 안내](https://github.com/huggingface/lerobot/blob/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/docs/source/robotwin.mdx)

## 소스형 벤치마크 자동 준비

LIBERO-plus, RoboCasa365, VLABench도 평가 실행을 처음 확정할 때 고정된 상위 소스를 사용해 준비할 수 있다. 선택한 호환 LeRobot Conda 환경을 관리자 전용 위치에 복제하고, 시뮬레이터 의존성과 자산을 설치한 뒤 네이티브 LeRobot의 초기화·행동 적용·물리 계산 경로를 검사한다. 이 과정에서는 정책 가중치를 다운로드하지 않는다. 모델 체크포인트는 작업 설정에서 별도로 선택한다.

| 벤치마크 | 준비하는 환경과 자산 |
| --- | --- |
| LIBERO-plus | 정확한 포크와 robosuite 1.4.1, 전용 LIBERO 설정, 새 환경 내부의 ImageMagick·fontconfig, 공식 섭동 자산 6.40 GB를 준비하고 게시된 해시와 대조한다. 복제 환경의 `hf-libero`만 포크로 교체하며 원래 Conda 환경은 보존한다. |
| RoboCasa365 | 고정된 RoboCasa·robosuite 소스, 호환 수치 연산 라이브러리, 주방 텍스처·설비와 선택한 물체 자산을 준비한다. 기본은 `lightwheel`이며 `--env.obj_registries`로 다른 자산을 명시하면 해당 팩도 받는다. 상위 소스의 패키지 정보는 그대로 두고 소스를 직접 읽어 오래된 `lerobot==0.3.3` 설치 요구를 피한다. |
| VLABench | 고정된 VLABench·경로 계획 소스, MuJoCo 3.2.2, dm_control 1.0.22, Open3D, Rtree 1.4.1과 공식 장면·물체 자산을 준비한다. Google Drive의 큰 파일은 HTTP(Hypertext Transfer Protocol, 하이퍼텍스트 전송 규약) 범위 요청으로 나누어 받고 보존된 부분 파일에서 재개한다. 압축 파일은 합계 6.00 GB, 해제한 자산은 약 12 GB이며 환경과 압축 해제에 필요한 추가 공간이 필요하다. |

관리자 전용 환경은 프로젝트 내부 `externals/.runtimes/`에 저장한다. 다른 디스크를 사용하려면 `CONVLEX_RUNTIME_CACHE`에 절대 경로를 지정한다. 캐시에는 소스 버전, 자산 해시, 의존성 조건과 검증된 환경을 기록한다. 준비 도중 중단된 환경은 사용 가능 상태로 게시하지 않는다. 정상 수신한 압축 파일은 전체 환경의 패키지·물리 검증이 끝날 때까지 재시도에 사용하고, 사용 가능 상태로 게시한 직후 다운로드 폴더를 삭제한다. 다음 실행에서는 호환 환경을 재사용하고 실제 물리 계산 경로를 다시 검사한다. 기존 LeRobot 소스, 사용자 데이터셋·체크포인트와 원래 Conda 환경은 보존한다.

관리자 전용 소스 환경에서도 선택한 정책 환경의 Hugging Face·Torch 모델 캐시 경로를 유지한다. 인증과 지원되는 실행 변수는 활성화된 자식 프로세스의 메모리에서 이어받고, 비활성화할 때 이전 값으로 복원한다. 인증 값은 작업 설정·환경 검증 기록·활성화 스크립트에 복사하지 않는다. 시뮬레이터와 컴파일러의 전용 경로를 우선한다.

Linux x86-64와 LeRobot 0.6.2에서 네 소스 벤치마크 모두 새 환경 준비·실제 카메라·초기화·물리 한 단계·환경 재사용을 검증했다. RoboTwin은 임시 초기화 정책으로 배치 크기 2의 동기 작업 두 개와 비동기 작업 한 개, 합계 짧은 에피소드 여섯 개를 실행했다. 스크립트·로그 저장과 부모 프로세스의 예기치 않은 종료에 따른 자식 정리도 확인했다. 모델 성능은 학습된 체크포인트와 별도의 평가 기준으로 측정한다.

개발 검증에서는 `CONVLEX_RUNTIME_CACHE`와 작업 결과를 전용 임시 경로로 지정하고 확인 직후 해당 경로만 정리한다. 실제 사용을 위해 남겨 둔 시뮬레이터 환경과 자산은 이후 실행에도 필요한 의존성이다. 이를 지우면 다음 실행에서 다시 준비한다. 드라이버 누락, 하드웨어 제한, 상위 다운로드 권한 또는 소스 버전 문제가 있으면 사용 가능으로 처리하지 않고 원인과 설치 로그를 보여준다.
