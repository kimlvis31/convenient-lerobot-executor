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
conda run -n lerobot python -m pip install --no-deps --no-build-isolation \
  "./externals/TinyVLA/lerobot_policy_tinyvla"
```

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

작업을 시작하면 convLexecutor가 선택한 모델과 벤치마크를 함께 실행할 수 있는 Conda 환경을 찾아 재사용합니다. π 계열의 SentencePiece처럼 현재 LeRobot 버전이 선언한 패키지가 빠져 있으면 필요한 것만 설치합니다. 기존 버전을 유지한 상태에서 의존성이 충돌하면 별도 환경을 만듭니다. 미리보기에서는 설치하지 않습니다. **Manual Setup**으로 표시된 벤치마크는 별도 시뮬레이터 설정이 필요하며, 설치된 LeRobot 환경 플러그인은 자동으로 조회합니다. 체크포인트는 작업 설정에서 선택합니다. PushT와 ALOHA는 작업 하나당 태스크 하나를 평가합니다. 배치 추론은 유지하고 시뮬레이터 단계는 기본적으로 동기 실행합니다. Meta-World는 시뮬레이터의 기본 에피소드 길이를 사용합니다.

정책이나 그 모델이 사용하는 하위 모델에 승인이 필요하면 Settings에서 알려진 저장소 링크를 볼 수 있습니다. 예를 들어 GR00T N1.7은 정책 체크포인트가 공개 상태여도 Cosmos-Reason2-2B 접근 권한을 요구할 수 있습니다. 실행 컴퓨터에서 쓰는 계정으로 안내된 저장소를 열어 약관에 동의하거나 권한을 신청하세요. 승인 후 해당 저장소를 읽을 수 있는 토큰으로 다시 실행하면 됩니다. 실행 도중 새로 발견된 저장소의 권한 오류는 Result에 표시됩니다. [Hugging Face 모델 접근 안내](https://huggingface.co/docs/hub/models-gated)

로컬 학습 데이터셋은 폴더를 선택하면 됩니다. `selection.json`에 저장소 이름과 학습 에피소드 목록이 있으면 자동으로 사용합니다. TinyVLA는 로컬 학습 분할의 태스크가 선택한 태스크와 일치하면 해당 분할을 자동으로 사용합니다. **Use a dataset repository**를 고르면 Hugging Face 저장소 이름을 입력합니다.

정책 목록은 로컬 체크포인트가 없는 모델을 흐리게 표시합니다. Hugging Face 캐시도 로컬에 포함합니다. 평가에는 선택한 벤치마크에 맞는 체크포인트를 사용합니다. LeRobot 모델은 현재 버전의 정책 설정이 지원하는 학습 범위와 LoRA(Low-Rank Adaptation, 저랭크 적응), 학습 옵션을 보여줍니다. TinyVLA는 행동 헤드만 학습하거나 행동 헤드와 언어 어텐션에 LoRA(Low-Rank Adaptation, 저랭크 적응)를 함께 적용할 수 있습니다. 정확한 재개는 기존 설정과 옵티마이저 상태를 복원합니다.

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

RoboTwin 평가에는 상위 시뮬레이터 소스, 로봇·물체 자산, 네이티브 라이브러리를 먼저 준비한다. `ROBOTWIN_ROOT`로 해당 저장소를 지정하거나 선택한 환경에서 `envs` 패키지를 찾을 수 있게 설정한다. 실행기는 필수 파일과 MPlib·NumPy 호환성을 검사하고, 확인된 카메라 이름을 연결하며, 시뮬레이터 자산 루트에서 평가를 실행한다. 최초 소스·시스템 라이브러리 설치는 [검증한 LeRobot 버전의 상위 설치 안내](https://github.com/huggingface/lerobot/blob/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/docs/source/robotwin.mdx)를 따른다. 이번 검증은 LeRobot 소스와 설치된 패키지 정보 및 NumPy·PyArrow·Transformers·네이티브 라이브러리 버전을 맞춘 별도 컨테이너에서 진행했다.
