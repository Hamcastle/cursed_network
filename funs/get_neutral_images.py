import os
import sys
import glob
import shutil
import itertools
import random
import numpy as np

from tqdm import tqdm

def main():
	'''
	Simple function to take a random subset of LabelMe images as the "neutral" dataset for the cursedness
	detector
	'''
	try:
		root_dir = os.path.expanduser('~/Sync/projects/work_projects/experiments/semantic_scene_stats/')
		source_image_dir = root_dir + 'data/images/'
		output_dir = os.path.expanduser('~/Sync/projects/personal_projects/cursed_image_detection/data/neutral/')

		#Get the number of cursed/blessed images and select a sample equal to the larger set size
		cursed_image_count = len(glob.glob(os.path.expanduser('~/Sync/projects/personal_projects/cursed_image_detection/data/reddit_sub_cursedimages/*')))+len(
			glob.glob(os.path.expanduser('~/Sync/projects/personal_projects/cursed_image_detection/data/twitter_account_cursedimages/*')))
		blessed_image_count = len(glob.glob(os.path.expanduser('~/Sync/projects/personal_projects/cursed_image_detection/data/reddit_sub_Blessed_Images/*')))+len(
			glob.glob(os.path.expanduser('~/Sync/projects/personal_projects/cursed_image_detection/data/twitter_account_blessediimages/*')))

		if cursed_image_count > blessed_image_count:
			neutral_count = cursed_image_count
		elif cursed_image_count < blessed_image_count:
			neutral_count = blessed_image_count

		image_filenames = []
		for root, dirs, files in os.walk(os.path.expanduser('~/Sync/projects/work_projects/experiments/semantic_scene_stats/data/images/'), topdown=False):
			for name in files:
				image_filenames.append(os.path.join(root, name))

		sorted_image_filenames = sorted(image_filenames)
	    
	    #Take a random subset of 100 images to work with
		random.seed(631989)
		indices = random.sample(range(len(image_filenames)), neutral_count)
		selected_images = [sorted_image_filenames[i] for i in indices]

		#Copy each neutral image in this list to a folder
		for each_image in tqdm(selected_images):
			shutil.copy(each_image,output_dir)
	except:
		raise

if __name__ == '__main__':
	main()