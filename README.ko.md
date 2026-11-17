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
python -B ~/convLexecutor/scripts/prepare_examples.py --dry-run
python -B ~/convLexecutor/scripts/prepare_examples.py
python -B ~/convLexecutor/executor/manage.py --inventory
python -B ~/convLexecutor/executor/manage.py \
  --queue ~/convLexecutor/scripts/example_script.py
```

준비 스크립트는 버전을 고정한 공개 기반 가중치와 크림치즈 집기 시연 1개를 사용하는 데 필요한 파일을 받습니다. 행동 출력부를 4단계 학습해 `outputs/example/`에 4단계 모델과 2단계 재개 상태를 만듭니다. 재개 상태에는 해당 컴퓨터의 기반 모델 위치가 들어가므로 각 컴퓨터에서 준비 명령을 실행합니다.

직접 설정하려면 `--queue` 없이 시작합니다. **Import a job queue script**를 선택하면 예제 파일을 고를 수 있습니다. SmolVLA를 사용할 때는 `"adapter": "native"`, `"model": "smolvla"`를 지정합니다. [예제 설정](scripts/example_script.py)

각 작업의 실제 실행 파일 `run.py`, 다시 가져올 수 있는 `script.py`, 시작·종료·소요 시간과 콘솔 로그는 `executor/log/<실행명>/`에 기록합니다. 평가 결과는 `outputs/evaluation/`, 학습 결과는 `outputs/training/`에 저장합니다. Git으로 관리하는 파일은 소스, 설정과 출처 목록입니다. 데이터, 가중치, 실행 로그와 결과는 로컬에서 생성합니다.

설치 작업을 에이전트에게 맡길 때는 [AGENTS.md](AGENTS.md)를 사용하세요.

## 라이선스

프로젝트 자체 코드는 MIT(Massachusetts Institute of Technology, 매사추세츠 공과대학) 라이선스를 따릅니다. 외부 소스와 공개 데이터는 각각의 원래 라이선스를 따릅니다. [프로젝트 라이선스](LICENSE) · [외부 소스와 라이선스](THIRD_PARTY.md)
