from typing import Any, Iterator, Tuple
import glob
import os
import json
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import logging
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Ssv2(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Something-Something V2 dataset in RLDS format."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release of Something-Something V2 in RLDS format.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation, etc.)."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "RLDS-converted Something-Something V2 dataset. "
                "SSv2 contains labeled video clips demonstrating basic human-object interactions."
            ),
            features=tfds.features.FeaturesDict({
                'episode_metadata': tfds.features.FeaturesDict({
                    'video_id': tf.string,    # Unique identifier for each video
                    'label': tf.string,       # SSv2 action label
                    'file_path': tf.string,   # Path to the original video file
                }),
                'steps': tfds.features.Sequence({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(shape=(224, 224, 3)),  # Extracted video frames
                        # 'wrist_image': tfds.features.Image(shape=(224, 224, 3)),  # Optional, commented out
                        # 'state': tfds.features.Text(),  # Optional, commented out
                    }),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc="Discount if provided, default to 1.",
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc="Reward if provided, 1 on final step for demos.",
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True on first step of the episode."
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True on last step of the episode."
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True on last step of the episode if it is a terminal step, True for demos.",
                    ),
                    'language_instruction': tfds.features.Text(
                        doc="Language Instruction."
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=tf.float32,
                        doc="Kona language embedding. "
                            "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                    ),
                }),
            }),
            supervised_keys=None,
            homepage="https://github.com/20bn/something-something",
            citation="""@article{goyal2017something,
                      title={Something-Something V2: A Large Dataset for Fine-grained Action Recognition},
                      author={Goyal, Ram and others},
                      journal={arXiv preprint arXiv:1803.08634},
                      year={2018}
                    }""",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        data_dir = os.path.expanduser('~/ssv2/')            # Path to .mp4 files
        labels_dir = os.path.join(data_dir, 'labels/')      # Path to label files

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'split': 'train',
                    'data_dir': data_dir,
                    'label_file': os.path.join(labels_dir, 'train.txt'),
                    'limit': 1000,  # test with only 1000 first
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    'split': 'validation',
                    'data_dir': data_dir,
                    'label_file': os.path.join(labels_dir, 'validation.txt'),
                    'limit': 200,  # test with only 200 first
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    'split': 'test',
                    'data_dir': data_dir,
                    'label_file': os.path.join(labels_dir, 'test-answers.csv'),
                    'limit': 100,  # test with onl 100 first
                },
            ),
        ]

    @lru_cache(maxsize=None)
    def get_language_embedding(self, label: str) -> np.ndarray:
        """Retrieve language embedding, using cache to avoid recomputation."""
        embedding = self._embed([label])[0].numpy()
        assert embedding.shape == (512,), f"Unexpected language_embedding shape: {embedding.shape}"
        return embedding.astype(np.float32)

    def _generate_examples(self, split: str, data_dir: str, label_file: str, limit: int = None) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        # Load labels.json to map label_ids to label_strings
        labels_json_path = os.path.join(os.path.dirname(label_file), 'labels.json')
        if not os.path.exists(labels_json_path):
            logger.error(f"labels.json not found at {labels_json_path}.")
            return  # Exit generator

        with open(labels_json_path, 'r') as file:
            labels = json.load(file)
        
        # Invert labels.json to map label_id to label_string
        label_id_to_string = {v: k for k, v in labels.items()}

        # Load label mappings from the label file
        label_map = {}
        if split != 'test':
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        video_id, label_id = parts[0], parts[1]
                        label_string = label_id_to_string.get(label_id, "unknown")
                        label_map[video_id] = label_string
                    else:
                        logger.warning(f"Invalid line format in {label_file}: {line}")
        else:
            # For test split, read from test-answers.csv
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(';')
                    if len(parts) == 2:
                        video_id, label_string = parts[0], parts[1]
                        label_map[video_id] = label_string
                    else:
                        logger.warning(f"Invalid line format in {label_file}: {line}")

        # Find all .mp4 files in the data directory
        video_files = glob.glob(os.path.join(data_dir, '*.mp4'))

        processed = 0
        skipped = 0

        for video_file in video_files:
            if limit is not None and processed >= limit:
                break
            try:
                video_id = os.path.splitext(os.path.basename(video_file))[0]
                label = label_map.get(video_id, "unknown" if split == 'test' else None)

                if label is None:
                    logger.warning(f"No label found for video {video_id} in split '{split}'. Assigning 'unknown'.")
                    label = "unknown"

                frames = self._extract_frames(video_file)

                if not frames:
                    logger.warning(f"No frames extracted from {video_file}. Skipping.")
                    skipped += 1
                    continue

                # Compute language embedding
                lang_embedding = self.get_language_embedding(label)

                # Assemble episode steps
                episode_steps = []
                for i, frame in enumerate(frames):
                    step_data = {
                        'observation': {
                            'image': frame,
                        },
                        'discount': 1.0,
                        'reward': float(i == (len(frames) - 1)),
                        'is_first': i == 0,
                        'is_last': i == (len(frames) - 1),
                        'is_terminal': i == (len(frames) - 1),
                        'language_instruction': label,
                        'language_embedding': lang_embedding,  # 512-dimensional vector
                    }
                    episode_steps.append(step_data)

                # Create output sample
                sample = {
                    'steps': episode_steps,
                    'episode_metadata': {
                        'video_id': video_id,
                        'label': label,
                        'file_path': video_file,
                    },
                }

                yield video_id, sample

                processed += 1

                if processed % 1000 == 0:
                    logger.info(f"Processed {processed} videos in split '{split}'.")

            except Exception as e:
                logger.warning(f"Failed to process {video_file}: {e}")
                skipped += 1
                continue

        logger.info(f"Finished processing split '{split}': {processed} processed, {skipped} skipped.")

    def _extract_frames(self, video_path: str, resize: Tuple[int, int] = (224, 224)) -> list:
        """Extract frames from a video file.

        Args:
            video_path (str): Path to the video file.
            resize (tuple): Desired frame size.

        Returns:
            list: List of processed frames.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                # Resize frame
                frame = cv2.resize(frame, resize)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            except Exception as e:
                logger.warning(f"Failed to process frame {frame_count} in {video_path}: {e}")
            frame_count += 1
        cap.release()
        return frames
