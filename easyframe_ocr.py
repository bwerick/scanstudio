import easyocr
reader = easyocr.Reader(['en'], detection='DB', recognition = 'Transformer') # this needs to run only once to load the model into memory
result = reader.readtext('/Users/erickduarte/git/segmentation/test_frames/The Slow Regard of Silent Things - Patrick Rothfuss (GoPro).mov/0000002070.jpg', detail = 0)

print(result)