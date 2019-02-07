
#Setup Paths

DATA = data/
FUNS = funs/
OUT  = out/
FIGS = figs/

# Targets
setup_folder_structure:
	mkdir $(DATA)
	mkdir $(DATA)/neutral/
	mkdir $(DATA)/train/
	mkdir $(DATA)/validation/
	mkdir $(DATA)/test/
	mkdir $(DATA)/train/cursed/
	mkdir $(DATA)/train/neutral/
	mkdir $(DATA)/validation/cursed/
	mkdir $(DATA)/validation/neutral/
	mkdir $(DATA)/test/cursed/
	mkdir $(DATA)/test/neutral/
	mkdir $(FUNS)
	mkdir $(OUT)
	mkdir $(FIGS)

get_rip_me:
	wget https://github.com/RipMeApp/ripme/releases/download/1.7.77/ripme.jar -P $(FUNS)/ripme

get_cursed_images:
	java -jar $(FUNS)/ripme/ripme.jar -u "https://twitter.com/cursedimages?lang=en" -l $(DATA)/
	java -jar $(FUNS)/ripme/ripme.jar -u "https://www.reddit.com/r/cursedimages/"   -l $(DATA)/

move_images_to_correct_subfolders: 
	python $(FUNS)/move_images_to_correct_subfolders.py

train_model:
	python $(FUNS)/train_cursed_image_model.py


cleanup_data_dir:
	rm -rf $(DATA)/cursed/
	rm -rf $(DATA)/neutral/
	rm -rf $(DATA)/reddit_sub_cursedimages/
	rm -rf $(DATA)/twitter_account_cursedimages/
