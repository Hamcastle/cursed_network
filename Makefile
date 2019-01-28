#switch for matlab path based on OS
OS := $(shell uname)
ifeq ($(OS),Darwin)
	MATLAB = /Applications/MATLAB_R2015b.app/bin/matlab
	PYTHON = $(HOME)/anaconda/envs/cursed_image/bin/python
else
	MATLAB = /home/dylanrose/matlab/bin/matlab
	PYTHON = $(HOME)/anaconda3/envs/cursed_image_linux/bin/python
endif

#Setup Paths

DATA = $(HOME)/Sync/projects/personal_projects/cursed_image_detection/data
FUNS = $(HOME)/Sync/projects/personal_projects/cursed_image_detection/funs
OUT  = $(HOME)/Sync/projects/personal_projects/cursed_image_detection/out
FIGS = $(HOME)/Sync/projects/personal_projects/cursed_image_detection/figs

# Targets
setup_folder_structure:
	mkdir $(DATA)
	mkdir $(DATA)/neutral/
	mkdir $(DATA)/train/
	mkdir $(DATA)/validation/
	mkdir $(DATA)/test/
	mkdir $(DATA)/train/cursed/
	mkdir $(DATA)/train/blessed/
	mkdir $(DATA)/train/neutral/
	mkdir $(DATA)/validation/cursed/
	mkdir $(DATA)/validation/blessed/
	mkdir $(DATA)/validation/neutral/
	mkdir $(DATA)/test/cursed/
	mkdir $(DATA)/test/blessed/
	mkdir $(DATA)/test/netural/
	mkdir $(FUNS)
	mkdir $(OUT)
	mkdir $(FIGS)

get_rip_me:
	wget https://github.com/RipMeApp/ripme/releases/download/1.7.77/ripme.jar -P $(FUNS)/ripme

get_cursed_images:
	java -jar $(FUNS)/ripme/ripme.jar -u "https://twitter.com/cursedimages?lang=en" -l $(DATA)/
	java -jar $(FUNS)/ripme/ripme.jar -u "https://www.reddit.com/r/cursedimages/"   -l $(DATA)/

get_blessed_images:
	java -jar $(FUNS)/ripme/ripme.jar -u "https://twitter.com/blessediimages?lang=en" -l $(DATA)/
	java -jar $(FUNS)/ripme/ripme.jar -u "https://www.reddit.com/r/Blessed_Images/" -l $(DATA)/

get_neutral_images:
	$(PYTHON) $(FUNS)/get_neutral_images.py
