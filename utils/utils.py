import argparse
import yaml
from dotmap import DotMap
from functools import reduce
from operator import getitem
from distutils.util import strtobool
from pathlib import Path

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

def parse_config(config_file_path: str) -> DotMap:
    """YAML 설정 파일을 파싱합니다."""
    # 파일을 UTF-8 인코딩으로 엽니다.
    with open(config_file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return DotMap(config, _dynamic=False)

def parse_command_line_args(config: DotMap) -> DotMap:
    """커맨드 라인 인수를 파싱합니다."""
    parser = argparse.ArgumentParser()

    # 설정 구조에 따라 자동으로 커맨드라인 인수를 추가합니다.
    def add_arguments(section, prefix=''):
        for key, value in section.items():
            full_key = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                add_arguments(value, prefix=full_key)
            else:
                # 리스트 값인지 확인
                if isinstance(value, list):
                    # 리스트를 콤마로 구분된 문자열로 변환
                    parser.add_argument(f'--{full_key}', default=value, type=type(value[0]), nargs='+', help=f'{full_key}의 값')
                else:
                    if type(value) == bool:
                        parser.add_argument(f'--{full_key}', default=value, type=strtobool, help=f'{full_key}의 값')
                    else:
                        parser.add_argument(f'--{full_key}', default=value, type=type(value), help=f'{full_key}의 값')

    add_arguments(config)

    args, _ = parser.parse_known_args()
    args = DotMap(vars(args), _dynamic=False)
    return args

def merge_configs(config: DotMap, args: DotMap) -> DotMap:
    """커맨드 라인 인수를 설정에 병합합니다. 커맨드 라인 인수는 설정 파일보다 우선합니다."""
    keys_to_modify = []

    def update_config(config, key, value):
        *keys, last_key = key.split('.')
        reduce(getitem, keys, config)[last_key] = value

    # 재귀적으로 커맨드라인 인수를 설정에 병합
    def get_updates(section, args, prefix=''):
        for key, value in section.items():
            full_key = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                get_updates(value, args, prefix=full_key)
            else:
                # 커맨드라인 인수가 설정 파일 값과 다르면 병합 대상에 추가
                if hasattr(args, full_key) and getattr(args, full_key) is not None:
                    keys_to_modify.append((full_key, getattr(args, full_key)))

    # 설정 병합 시작
    get_updates(config, args)

    for key, value in keys_to_modify:
        update_config(config, key, value)

    # experiment_name 기본값 설정
    if not hasattr(args, 'ARNIQA_Experiment') or not args.experiment_name:
        config.experiment_name = config.get('ARNIQA_Experiment', 'default_experiment')

    return config
