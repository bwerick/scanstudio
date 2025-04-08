import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
result = reader.readtext('/Users/erickduarte/git/segmentation/test_frames/The Plague Recording.mov/0000000315.jpg', detail = 0)

print(result)