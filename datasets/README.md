# Dataset creation
If you want to run FaceLIBRAS with a custom dataset, follow the instructions:

- Create a folder in you system
- Add all your images inside this folder
- Create a info.txt file
- For each image in the folder add a line in info.txt with IMAGE_NAME.EXT@CLASS

# Important
- The CLASS variable is a number 'zero-based'
- Always use @ as separator
- The image need to be 100x100 in size to fit the model
- The image is resized to 100x100 at runtime
