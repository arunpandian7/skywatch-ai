import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import skywatchai.SkywatchAI as skai
import skywatchai.SkywatchDB as skdb
import numpy as np

img_dir = 'test/images/'

print("Running SkywatchAI Unit Tests...\n")
print("Test (1/4):")
print("SkywatchAI Compare Function \n")
passed = 0

# Expected Output True 
test1_image1 = img_dir+'image1.jpg'
test1_image2 = img_dir+'image2.jpg'
test1_image3 = img_dir + 'image3.jpg'



print(f"Input Images : {test1_image1} and {test1_image2}")
print("Expected Output : True")
result1 = skai.compare(test1_image1, test1_image2)
print("SkywatchAI Result :",result1)

print(f"\nInput Images : {test1_image1} and {test1_image3}")
print("Expected Output : False")
result2 = skai.compare(test1_image1, test1_image3)
print("SkywatchAI Result :",result2)

if result1==True and result2==False:
    passed += 1
    print("Test 1 Passed...")
else:
    print("Test 1 Failed...")
print("\n------------------------------\n")


print("Test (2/4):")
print("SkywatchAI Face Detection Pipeline")

test2_image = img_dir + "oscar.jpg"

print(f"Input Image : {test2_image}")
detections = skai.get_faces(test2_image)
detected_img = skai.detect_faces(test2_image)

if len(detections) == 11 and type(detected_img) == np.ndarray:
    passed += 1
    print("Test 2 Passed...")

else:
    print("Test 2 Failed...")
print("\n------------------------------\n")

print("Test (3/4):")
print("SkywatchDB Building and Loading")

try:
    skdb.build_db('database/people', 'database/')
    faceDB, nameMap = skdb.load_db(path='database/')
    print("\nTest 3 Passed...")
    passed+=1
except:
    print("\nTest 3 Failed...")
    print(f"Cannot Continue..\n Tests Passed:{passed}/5 \nExiting Testing...")
    exit()
print("\n------------------------------\n")


print("Test (4/4):")
print("SkywatchAI Face Recognition")

test4_image = img_dir + 'braniston.jpg'
annotImage = skai.find_people(test4_image, faceDB, nameMap)
names = skai.get_names(test4_image, faceDB, nameMap)

print("Input Image", test4_image)
print("Expected Output: ['Brad Pitt', 'Jennifer Aniston', 'Unidentified']")
print("Output:", names)
if names == ['Brad Pitt', 'Jennifer Aniston', 'Unidentified'] and type(annotImage) == np.ndarray:
    print("\nTest 4 Passed...")
    passed+=1
else:
    print("\nTest 4 Failed...")
print("\n------------------------------\n")
 

print("\n","="*30)
print(f"Test Passed : ({passed}/4)")
