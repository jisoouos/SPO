from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class TrainItem:
    path: str
    speaker: str
    label: int


@dataclass
class EnrollmentItem:
    key: int
    path: str


@dataclass
class TestTrial:
    key1: int
    key2: int
    label: int


class SVTrainDB:
    def __init__(self, train_samples: str, trial_list: dict):
        self.train_set: List[List[TrainItem]] = []

        with open(train_samples, "r") as f:
            for line in f:
                spk, label_str, wav_path = line.strip().split()
                label = int(label_str)
                item = TrainItem(path=wav_path, speaker=spk, label=label)
                if len(self.train_set) <= label:
                    self.train_set.extend([[] for _ in range(label - len(self.train_set) + 1)])
                self.train_set[label].append(item)

        self.trials: Dict[str, List[TestTrial]] = {}
        self.enrollment_samples: List[EnrollmentItem] = []
        # map per trial file: raw_key -> global_key
        self.key_map: Dict[str, Dict[int, int]] = {}

        next_key = 0
        enrollment_by_new_key: Dict[int, EnrollmentItem] = {}
        path_to_global_key: Dict[str, int] = {}

        for trial_name, trial_path in trial_list.items():
            trials_raw, samples_raw = self.parse_trials(trial_path)

            raw_to_global: Dict[int, int] = {}
            for sample in samples_raw:
                if sample.path in path_to_global_key:
                    global_key = path_to_global_key[sample.path]
                else:
                    global_key = next_key
                    path_to_global_key[sample.path] = global_key
                    enrollment_by_new_key[global_key] = EnrollmentItem(
                        key=global_key, path=sample.path
                    )
                    next_key += 1
                raw_to_global[sample.key] = global_key

            remapped_trials: List[TestTrial] = []
            for trial in trials_raw:
                try:
                    key1 = raw_to_global[trial.key1]
                    key2 = raw_to_global[trial.key2]
                except KeyError as e:
                    raise ValueError(
                        f"Trial references unknown key {e.args[0]} in {trial_path}"
                    ) from e
                remapped_trials.append(TestTrial(key1=key1, key2=key2, label=trial.label))

            self.trials[trial_name] = remapped_trials
            self.key_map[trial_name] = raw_to_global

        self.enrollment_samples = [enrollment_by_new_key[k] for k in sorted(enrollment_by_new_key)]

    @staticmethod
    def parse_trials(path: str) -> Tuple[List[TestTrial], List[EnrollmentItem]]:
        trials: List[TestTrial] = []
        samples: List[EnrollmentItem] = []
        seen_keys: Dict[int, str] = {}

        with open(path, "r") as f:
            for line in f:
                label_str, key1_str, key2_str, sample1, sample2 = line.strip().split()
                label = int(label_str)
                key1 = int(key1_str)
                key2 = int(key2_str)

                trials.append(TestTrial(key1=key1, key2=key2, label=label))

                for key, sample in ((key1, sample1), (key2, sample2)):
                    if key not in seen_keys:
                        samples.append(EnrollmentItem(key=key, path=sample))
                        seen_keys[key] = sample
                    elif seen_keys[key] != sample:
                        raise ValueError(
                            f"Duplicate enrollment key {key} maps to both "
                            f"'{seen_keys[key]}' and '{sample}' in {path}"
                        )

        return trials, samples
