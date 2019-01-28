import os
import glob
import shutil
import random

from tqdm import tqdm

def main():
	'''
	Function to move images around to correct subfolders
	'''
	try:
		
		#Get lists of paths to the image files
		cursed_image_list= glob.glob(os.path.expanduser('~/Sync/projects/personal_projects/cursed_image_detection/data/reddit_sub_cursedimages/*')
			)+glob.glob(os.path.expanduser('~/Sync/projects/personal_projects/cursed_image_detection/data/twitter_account_cursedimages/*'))
		blessed_image_list = glob.glob(os.path.expanduser('~/Sync/projects/personal_projects/cursed_image_detection/data/reddit_sub_Blessed_Images/*')
			)+glob.glob(os.path.expanduser('~/Sync/projects/personal_projects/cursed_image_detection/data/twitter_account_blessediimages/*'))
		neutral_image_list  = glob.glob(os.path.expanduser('~/Sync/projects/personal_projects/cursed_image_detection/data/neutral/*'))

		# Going with a 70/20/remainder up to 10% train/val/test/ split
		random.seed(324823)
		all_images_list = [cursed_image_list,blessed_image_list,neutral_image_list]
		for each_list_idx,each_list in tqdm(enumerate(all_images_list)):
			random.shuffle(each_list)
			num_images_for_training  = int(round(len(each_list)*0.7))
			images_for_val_test      = each_list[num_images_for_training:]
			images_for_training      = each_list[:num_images_for_training]
			num_images_for_val       = int(round(len(images_for_val_test)*0.66))
			images_for_test          = images_for_val_test[num_images_for_val:]
			images_for_val           = images_for_val_test[:num_images_for_val]
			
			#Crude: just set which type of image in the list you're currently working with on a pre-fixed order
			if each_list_idx == 0:
				image_type = 'cursed'
			elif each_list_idx == 1:
				image_type = 'blessed'
			else:
				image_type = 'neutral'

			#Destination paths
			train_path = os.path.expanduser('~/Sync/projects/personal_projects/cursed_image_detection/data/train/')+image_type+'/'
			val_path   = os.path.expanduser('~/Sync/projects/personal_projects/cursed_image_detection/data/validation/')+image_type+'/'
			test_path  = os.path.expanduser('~/Sync/projects/personal_projects/cursed_image_detection/data/test/')+image_type+'/'

			#Move all of the images into the appropriate subfolder
			[shutil.move(each_file,train_path) for each_file in images_for_training]
			[shutil.move(each_file,val_path) for each_file in images_for_val]
			[shutil.move(each_file,test_path) for each_file in images_for_test]

	except:
		raise

if __name__ == '__main__':
	main()