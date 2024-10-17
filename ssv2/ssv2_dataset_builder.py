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
        """Define data splits by invoking _generate_examples for each split."""
        base_dir = os.path.expanduser('~/ssv2/')  # Base directory containing all data

        # Return a dictionary mapping splits to generator calls
        return {
            'train': self._generate_examples(path=base_dir, split='train'),
            'val': self._generate_examples(path=base_dir, split='validation'),
            'test': self._generate_examples(path=base_dir, split='test'),
        }

    @lru_cache(maxsize=None)
    def get_language_embedding(self, label: str) -> np.ndarray:
        """Retrieve language embedding, using cache to avoid recomputation."""
        embedding = self._embed([label])[0].numpy()
        assert embedding.shape == (512,), f"Unexpected language_embedding shape: {embedding.shape}"
        return embedding.astype(np.float32)

    def _generate_examples(self, path: str, split: str) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        labels_json_path = os.path.join(path, 'labels', 'labels.json')

        if not os.path.exists(labels_json_path):
            logger.error(f"labels.json not found at {labels_json_path}.")
            return  # Exit generator

        with open(labels_json_path, 'r') as file:
            labels = json.load(file)
        
        # Invert labels.json to map label_id to label_string
        label_id_to_string = {v: k for k, v in labels.items()}

        # Determine label file based on split
        if split == 'train':
            label_file = os.path.join(path, 'labels', 'train.txt')
        elif split == 'validation':
            label_file = os.path.join(path, 'labels', 'validation.txt')
        elif split == 'test':
            label_file = os.path.join(path, 'labels', 'test-answers.csv')
        else:
            logger.error(f"Unknown split: {split}")
            return

        # Get video_ids for the current split
        video_ids = self._get_video_ids(label_file, split)

        for video_id, label_info in video_ids.items():
            video_filename = f"{video_id}.mp4"
            video_path = os.path.join(path, video_filename)

            # Determine label
            if split == 'test':
                # For test, label_map already contains label_string
                label_string = label_info
            else:
                # For train and validation, map label_id to label_string
                label_id = label_info
                label_string = label_id_to_string.get(label_id, "unknown")

            # Extract frames
            frames = self._extract_frames(video_path)

            # Compute language embedding
            lang_embedding = self.get_language_embedding(label_string)

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
                    'language_instruction': label_string,
                    'language_embedding': lang_embedding,  # 512-dimensional vector
                }
                episode_steps.append(step_data)

            # Create output sample
            sample = {
                'steps': episode_steps,
                'episode_metadata': {
                    'video_id': video_id,
                    'label': label_string,
                    'file_path': video_path,
                },
            }

            yield video_id, sample

    def _get_video_ids(self, label_file: str, split: str) -> dict:
        """
        Internal helper function to parse label files and return a mapping of video_id to label.
        
        Args:
            label_file (str): Path to the label file.
            split (str): Name of the split ('train', 'validation', 'test').

        Returns:
            dict: Mapping of video_id to label_id or label_string.
        """
        label_map = {}
        if not os.path.exists(label_file):
            logger.error(f"Label file does not exist: {label_file}")
            return label_map

        with open(label_file, 'r') as f:
            for line in f:
                if split == 'test':
                    # Test label files have format: video_id;label_string
                    parts = line.strip().split(';')
                    if len(parts) == 2:
                        video_id, label_string = parts
                        label_map[video_id] = label_string
                    else:
                        logger.warning(f"Invalid line format in {label_file}: {line.strip()}")
                else:
                    # Train and Validation label files have format: video_id label_id
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        video_id, label_id = parts[0], parts[1]
                        label_map[video_id] = label_id
                    else:
                        logger.warning(f"Invalid line format in {label_file}: {line.strip()}")

        return label_map

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
