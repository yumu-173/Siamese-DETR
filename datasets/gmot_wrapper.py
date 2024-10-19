import os.path as osp

import torch
from torch.utils.data import Dataset

from .mot_sequence import MOTSequence


class GMOT40Wrapper(Dataset):
	"""A Wrapper for the MOT_Sequence class to return multiple sequences."""

	def __init__(self):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""
		# mot_dir = '../gmot-main/data/COCO/'
		mot_dir = 'Dataset/gmot/'

		test_sequences = [
				'airplane-0', 
				'airplane-1', 
		    	'airplane-2', 
				'airplane-3', 
		        'ball-0', 
				'ball-1', 
				'ball-2', 
				'ball-3',
			    'balloon-0', 
				'balloon-1', 
				'balloon-2', 
				'balloon-3',
			    'bird-0', 
				'bird-1', 
				'bird-2', 
				'bird-3',
		        'boat-0', 
				'boat-1', 
				'boat-2', 
				'boat-3',
                'car-0', 
				'car-1', 
				'car-2', 
				'car-3',
                'fish-0', 
				'fish-1', 
				'fish-2', 
				'fish-3',
                'insect-0', 
				'insect-1', 
				'insect-2', 
				'insect-3',
                'person-0', 
				'person-1', 
				'person-2', 
				'person-3',
                'stock-0', 
				'stock-1', 
				'stock-2', 
				'stock-3',
				]


		sequences = test_sequences
		self._data = []
		for s in sequences:
			self._data.append(MOTSequence(f"{s}", mot_dir))

	def __len__(self):
		return len(self._data)

	def __getitem__(self, idx):
		return self._data[idx], self._data[idx].template



class GMOT40VisualizeAttentionWrapper(GMOT40Wrapper):

	def __init__(self):
		from .mot_sequence import MOTSequenceAttentionVisualization
		mot_dir = 'template/gmot_attention_visualization'
		test_sequences = [
			# 'attention-vis-1-cone',
			# 'attention-vis-1-car',
			# 'attention-vis-2-ball',
			# 'attention-vis-2-car',
			'attention-vis-2-person'
			]

		sequences = test_sequences
		self._data = []
		for s in sequences:
			self._data.append(MOTSequenceAttentionVisualization(f"{s}", mot_dir))